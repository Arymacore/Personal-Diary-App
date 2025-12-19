import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from textblob import TextBlob
from transformers import pipeline

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'diary.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
PER_PAGE = 7
def build_page_window(pagination, max_buttons=5):
    total = pagination.pages
    current = pagination.page

    if total <= max_buttons + 2:
        return list(range(1, total + 1))

    pages = [1]
    left = max(2, current - 2)
    right = min(total - 1, current + 2)

    if left > 2:
        pages.append('...')
    pages.extend(range(left, right + 1))
    if right < total - 1:
        pages.append('...')
    pages.append(total)

    return pages
MOOD_MAP = {
    "joyful": ("Joyful", "😀"),
    "sad": ("Sad", "😢"),
    "angry": ("Angry", "😡"),
    "fearful": ("Fearful", "😨"),
    "excited": ("Excited", "🤩"),
    "calm": ("Calm", "😐"),
    "neutral": ("Neutral", "⚪"),
}
# ========== 情绪模型：HuggingFace ==========

try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1  # 只要得分最高的那一个标签
    )
    print("✅ Emotion model loaded.")
except Exception as e:
    emotion_classifier = None
    print("⚠️ Could not load emotion model, fallback to TextBlob rules:", e)

def predict_mood(text):
    """
    情绪预测（升级版）：
    1. 优先使用 HuggingFace 情绪模型；
    2. 如果模型不可用 / 出错，则回退到 TextBlob + 关键词规则。
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    text_lower = text.lower()

    # 关键词表：后面既给 transformer 做细分，也给 fallback 用
    angry_words = ["angry", "mad", "furious", "irritated", "annoyed", "rage", "pissed"]
    fear_words = ["scared", "afraid", "fear", "terrified", "worried", "anxious", "nervous"]
    sad_words = ["sad", "depressed", "unhappy", "down", "miserable", "cry", "lonely"]
    joyful_words = ["happy", "joy", "delighted", "glad", "cheerful", "grateful"]
    excited_words = ["excited", "thrilled", "energetic", "pumped", "ecstatic", "hyped"]
    calm_words = ["calm", "relaxed", "peaceful", "chill", "okay", "fine"]

    # ------- 旧的规则版（用于回退） -------
    def rule_based():
        if any(w in text_lower for w in angry_words):
            return "angry"
        if any(w in text_lower for w in fear_words):
            return "fearful"
        if any(w in text_lower for w in excited_words):
            return "excited"
        if any(w in text_lower for w in sad_words):
            return "sad"
        if any(w in text_lower for w in joyful_words):
            return "joyful"
        if any(w in text_lower for w in calm_words):
            return "calm"

        if polarity > 0.3:
            return "joyful"
        elif polarity < -0.3:
            return "sad"
        else:
            return "neutral"

    # 空文本直接 Neutral
    if not text_lower.strip():
        return "neutral"

    # 如果模型没成功加载，直接用旧规则
    if emotion_classifier is None:
        return rule_based()

    # ------- 使用 HuggingFace 模型 -------
    try:
        preds = emotion_classifier(text)
        # 兼容不同返回格式：可能是 [ {label,score} ] 或 [ [ {..} ] ]
        if isinstance(preds, list):
            first = preds[0]
            if isinstance(first, list):
                first = first[0]
            label = first["label"].lower()
        else:
            label = preds["label"].lower()
    except Exception as e:
        print("Emotion model error:", e)
        return rule_based()

    # ------- 将 HuggingFace 标签映射到我们的 7 种情绪 -------
    # 模型标签：大概是 joy, anger, sadness, fear, neutral, surprise

    # 辅助：有没有强烈“负面词”
    has_negative_hint = (
            any(w in text_lower for w in sad_words)
            or any(w in text_lower for w in fear_words)
            or any(w in text_lower for w in angry_words)
    )

    # 🎯 特别照顾 calm：如果用户一直在强调 calm / relaxed，
    # 但整体情绪不是很强烈，就倾向于给 calm。
    def try_calm():
        if any(w in text_lower for w in calm_words) \
                and polarity > -0.1 and polarity < 0.5 \
                and not has_negative_hint:
            # 有 calm 词、没有明显负面、极性在 -0.1 ~ 0.5 之间 → 认为是 calm
            return "calm"
        return None

    # 1) joy
    if label == "joy":
        # 先看看能不能判成 calm（比如 "I'm really calm now."）
        calm_result = try_calm()
        if calm_result:
            return calm_result

        # 很开心、极端正向 → excited
        if any(w in text_lower for w in excited_words) or polarity > 0.6:
            return "excited"
        else:
            return "joyful"

    # 2) sadness
    if label == "sadness":
        return "sad"

    # 3) anger
    if label == "anger":
        return "angry"

    # 4) fear
    if label == "fear":
        return "fearful"

    # 5) neutral
    if label == "neutral":
        # neutral 里优先给 calm，但要确保没有强烈负面词
        calm_result = try_calm()
        if calm_result:
            return calm_result
        return "neutral"

    # 6) surprise
    if label == "surprise":
        # 惊喜 or 受惊吓：看一下极性
        if polarity >= 0:
            return "excited"
        else:
            return "fearful"

    # 兜底：遇到奇怪标签就退回规则版
    return rule_based()


def extract_tags(text):
    blob = TextBlob(text)
    nouns = []
    for word, pos in blob.tags:
        if pos.startswith('NN') and len(word) > 2:
            nouns.append(word.lower())
    unique = sorted(set(nouns))
    return ", ".join(unique[:5])


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    entries = db.relationship('Entry', backref='user', lazy=True)


class Entry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False, default="")
    text = db.Column(db.Text, nullable=False)
    tags = db.Column(db.String(200), nullable=True)
    mood = db.Column(db.String(50), nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose another.', 'danger')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    entries = Entry.query.filter_by(user_id=user.id).all()
    mood_counts = {}
    for e in entries:
        mood_counts[e.mood] = mood_counts.get(e.mood, 0) + 1
    pie_chart = None
    if entries:
        pie_chart = generate_pie_chart(mood_counts)
    return render_template('dashboard.html', entries=entries, pie_chart=pie_chart, username=user.username)


@app.route('/add_entry', methods=['GET', 'POST'])
def add_entry():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    from_dashboard = request.args.get('from_dashboard') == '1'
    if request.method == 'POST':
        title = request.form.get('title', '')
        text = request.form['text']
        date_str = request.form.get('date', '')
        if date_str:
            date_created = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
        else:
            date_created = datetime.utcnow()
        manual_tags = request.form.get('tags', '').strip()
        if manual_tags:
            tags = manual_tags
        else:
            tags = extract_tags(text)  # 用户没填 → 自动生成一份

        mood = predict_mood(text)
        new_entry = Entry(
            title=title,
            text=text,
            tags=tags,
            mood=mood,
            date_created=date_created,
            user_id=session['user_id']
        )
        db.session.add(new_entry)
        db.session.commit()
        flash('Entry added successfully! 🎉', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_entry.html', datetime=datetime, from_dashboard=from_dashboard)


@app.route('/view_entries')
def view_entries():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    page = request.args.get('page', 1, type=int)

    pagination = (
        Entry.query
        .filter_by(user_id=user.id)
        .order_by(Entry.date_created.desc())
        .paginate(page=page, per_page=PER_PAGE, error_out=False)
    )
    entries = pagination.items
    page_window = build_page_window(pagination)

    return render_template(
        'view_entries.html',
        entries=entries,
        pagination=pagination,
        page_window=page_window
    )
@app.route('/entry/<int:entry_id>')
def view_entry(entry_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    entry = Entry.query.get_or_404(entry_id)
    if entry.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('view_entries'))

    # 是否来自搜索页
    from_search = request.args.get('from_search') == '1'

    # 把搜索参数带回去，方便“Back to Search”
    search_params = {
        'keyword':   request.args.get('keyword', ''),
        'tags':      request.args.get('tags', ''),
        'mood':      request.args.get('mood', 'all'),
        'date_from': request.args.get('date_from', ''),
        'date_to':   request.args.get('date_to', ''),
        'page':      request.args.get('page', 1),
    }

    return render_template(
        'single_entry.html',     # ⚠️ 保持文件名 single_entry.html
        entry=entry,
        from_search=from_search,
        search_params=search_params
    )

@app.route('/search', methods=['GET'])
def search_entries():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    # ① 第一次打开 /search（地址栏里没有任何 ?xxx=）→ 不搜索，直接渲染空列表
    if not request.args:
        return render_template(
            'search.html',
            entries=[],
            pagination=None,
            page_window=[],
            keyword='',
            tag_text='',
            mood='all',          # HTML 里虽然写的是 "Choose mood"，value 还是 all
            date_from='',
            date_to='',
            has_searched=False   # 告诉模板：还没真正搜索
        )

    # ② 用户点了 Search（地址栏有参数了）→ 正常按条件搜索 + 分页
    keyword   = request.args.get('keyword', '').strip()
    tag_text  = request.args.get('tags', '').strip()
    mood      = request.args.get('mood', 'all')
    date_from = request.args.get('date_from', '')
    date_to   = request.args.get('date_to', '')

    query = Entry.query.filter_by(user_id=user.id)

    # 关键词
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(
            (Entry.title.ilike(like)) | (Entry.text.ilike(like))
        )

    # tags（可多个，用逗号）
    if tag_text:
        tags = [t.strip().lstrip('#') for t in tag_text.split(',') if t.strip()]
        for t in tags:
            query = query.filter(Entry.tags.ilike(f"%{t}%"))

    # 心情
    if mood and mood != 'all':
        query = query.filter(Entry.mood == mood)

    # 日期范围
    if date_from:
        try:
            start_dt = datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(Entry.date_created >= start_dt)
        except ValueError:
            pass

    if date_to:
        try:
            end_dt = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(Entry.date_created < end_dt)
        except ValueError:
            pass

    # 分页
    page = request.args.get('page', 1, type=int)
    pagination = query.order_by(Entry.date_created.desc()).paginate(
        page=page,
        per_page=PER_PAGE,
        error_out=False
    )
    entries = pagination.items
    page_window = build_page_window(pagination)

    return render_template(
        'search.html',
        entries=entries,
        pagination=pagination,
        page_window=page_window,
        keyword=keyword,
        tag_text=tag_text,
        mood=mood,
        date_from=date_from,
        date_to=date_to,
        has_searched=True     # 这次是真的搜过了
    )

@app.route('/edit_entry/<int:entry_id>', methods=['GET', 'POST'])
def edit_entry(entry_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    entry = Entry.query.get_or_404(entry_id)
    if entry.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('view_entries'))
    if request.method == 'POST':
        entry.title = request.form.get('title', entry.title)
        entry.text = request.form['text']
        date_str = request.form.get('date', '')
        if date_str:
            entry.date_created = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")

        manual_tags = request.form.get('tags', '').strip()
        if manual_tags:
            # 用户手动输入 → 完全按照用户的来
            entry.tags = manual_tags
        else:
            # 用户把 tags 清空了 → 帮他重新自动生成一份
            entry.tags = extract_tags(entry.text)

        # —— 新增：编辑时允许手动选择心情 ——
        mood_choice = request.form.get('mood_choice', 'auto')

        if mood_choice == 'auto':
            # 让模型根据最新文本重新判断
            entry.mood = predict_mood(entry.text)
        else:
            # 用户手动选了具体心情 → 直接覆盖
            entry.mood = mood_choice

        db.session.commit()
        flash('Entry updated!', 'success')
        return redirect(url_for('view_entries'))
    return render_template('edit_entry.html', entry=entry)


@app.route("/delete_entry/<int:entry_id>", methods=["POST"])
def delete_entry(entry_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    entry = Entry.query.get_or_404(entry_id)
    if entry.user_id != session["user_id"]:
        flash("Unauthorized access.", "danger")
        return redirect(url_for("view_entries"))
    db.session.delete(entry)
    db.session.commit()
    flash("Entry deleted successfully!", "success")
    return redirect(url_for("view_entries"))


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))


def generate_pie_chart(mood_counts):
    mood_color_map = {
        'joyful': '#7fd89e',
        'sad': '#4A5568',
        'angry': '#FF0000',
        'fearful': '#9b59b6',
        'excited': '#FFB6C1',
        'calm': '#8ac6ee',
        'neutral': '#bdc3c7',
        'other': '#dadada'
    }
    labels = []
    values = []
    colors = []
    for mood, count in mood_counts.items():
        labels.append(mood.title())
        values.append(count)
        colors.append(mood_color_map.get(mood, mood_color_map['other']))
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=140, pctdistance=0.77, textprops={'color': '#457458', 'fontsize': 13},
                                      wedgeprops={'linewidth': 2, 'edgecolor': '#fcfcfa'}, shadow=True)
    ax.axis('equal')
    plt.setp(autotexts, size=13, weight='bold', color="#333")
    plt.subplots_adjust(left=0.13, right=0.87, top=0.87, bottom=0.13)
    ax.set_title("Mood Distribution", fontsize=17, color="#457458", weight='bold', pad=16)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    return plot_data
@app.cli.command('recalc_mood')
def recalc_mood():
    """Recalculate mood for all entries based on current predict_mood()."""
    with app.app_context():
        entries = Entry.query.all()
        for e in entries:
            old = e.mood
            e.mood = predict_mood(e.text)
            print(f"Recalc id={e.id}: {old} -> {e.mood}")
        db.session.commit()
        print("Done: all moods recalculated.")

if __name__ == '__main__':
    app.run(debug=True)
