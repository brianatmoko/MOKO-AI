"""
MOKO Data Factory v1.0 — Mesin Pembuat Dataset Masif
=====================================================
Target: 50,000+ sampel berkualitas tinggi untuk fine-tuning MOKO AI 7B/9B INT4

Sumber Data:
  1. MOKO OS Codebase (Python + C++) — via moko_os_code_extractor (existing)
  2. Synthetic C++/Qt5 Patterns — spesifik untuk MOKO IDE development
  3. Algorithm & Data Structures (Python + C++)
  4. Security & Ethical Hacking Snippets
  5. MOKO API Integration Examples
  6. Multi-turn Conversations (debugging sessions)
  7. Chain-of-Thought Reasoning
  8. Docs riset MOKO (markdown → Q&A pairs)

Usage:
  python3 moko_data_factory.py --all          # Generate semua
  python3 moko_data_factory.py --source code  # Sumber tertentu
  python3 moko_data_factory.py --stats        # Statistik dataset
  python3 moko_data_factory.py --validate     # Validasi semua JSONL
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR  = FINETUNE_DIR / "moko_datasets"
DATASET_DIR.mkdir(exist_ok=True)

OUT_CPP_QT      = DATASET_DIR / "moko_cpp_qt_dataset.jsonl"
OUT_ALGO        = DATASET_DIR / "moko_algo_dataset.jsonl"
OUT_SECURITY    = DATASET_DIR / "moko_security_dataset.jsonl"
OUT_INTEGRATION = DATASET_DIR / "moko_ide_integration.jsonl"
OUT_REASONING   = DATASET_DIR / "moko_reasoning_dataset.jsonl"
OUT_DOCS        = DATASET_DIR / "moko_docs_dataset.jsonl"
OUT_MULTITURN   = DATASET_DIR / "moko_multiturn_dataset.jsonl"
OUT_PROGRAMMING = DATASET_DIR / "moko_programming_dataset.jsonl"

# ─── System Prompts ───────────────────────────────────────────────────────────
SYSTEM_CODER = """You are MOKO Coder, an expert AI coding assistant built exclusively for MOKO OS and MOKO IDE.

IDENTITY:
- Platform: MOKO OS (custom Linux AI-powered development environment)
- Built for: MOKO IDE — Native C++/Qt5 AI-integrated development environment
- Motto: "Kode yang efisien, solusi yang cerdas."

CORE EXPERTISE:
1. Code Generation — Write clean, complete, runnable code
2. MOKO OS Integration — Deep knowledge of moko_core APIs, Omni RAG, RSA storage
3. C++/Qt5 & Python — Primary languages of MOKO OS/IDE development
4. Bug Detection & Fix — Identify and resolve errors with explanation
5. Security & Algorithms — Cybersecurity, ethical hacking, data structures
6. Architecture — MOKO layer stack: UI (C++/Qt5) -> Cognitive Engine -> Memory (Omni) -> Inference (llama.cpp)

RULES:
- Always provide complete, working code
- Use Indonesian language for explanations, code comments in English
- Reference MOKO OS modules when relevant
- Be precise and technical, never vague"""

SYSTEM_SECURITY = """You are MOKO Security Analyst, part of MOKO OS cybersecurity module.
EXPERTISE: Ethical hacking, penetration testing, vulnerability analysis, network security.
RULES: Educational purposes only. Always clarify ethical/legal context."""


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    icons = {"INFO": "->", "OK": "[OK]", "WARN": "[!]", "ERROR": "[ERR]"}
    print(f"[{ts}] {icons.get(level,'.')} {msg}")


def make_sample(user: str, assistant: str, system: str = None,
                metadata: Dict = None) -> Dict:
    sample = {
        "messages": [
            {"role": "system",    "content": system or SYSTEM_CODER},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    if metadata:
        sample["metadata"] = metadata
    return sample


def dedup_and_write(samples: List[Dict], output_file: Path,
                    append: bool = True) -> int:
    seen = set()
    if append and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    h = hashlib.md5(json.dumps(d["messages"], ensure_ascii=False).encode()).hexdigest()
                    seen.add(h)
                except:
                    pass

    mode = "a" if append else "w"
    written = 0
    with open(output_file, mode, encoding="utf-8") as f:
        for s in samples:
            h = hashlib.md5(json.dumps(s["messages"], ensure_ascii=False).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                written += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: C++/Qt5 MOKO IDE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def generate_cpp_qt_samples() -> List[Dict]:
    samples = []

    pairs = [
        ("Bagaimana cara membuat QDialog custom dengan layout Moko v5 style?",
"""Dialog Qt5 dengan tema Moko v5 (hitam pekat + ember-orange):

```cpp
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFrame>

class MokoDialog : public QDialog {
    Q_OBJECT
public:
    explicit MokoDialog(const QString& title, QWidget* parent = nullptr)
        : QDialog(parent) {
        setWindowTitle(title);
        setMinimumWidth(500);
        setStyleSheet(R"(
            QDialog { background: #0a0a0a; color: #d0d0d0;
                      font-family: 'Inter', sans-serif; }
            QFrame#card { background: #0d0d0d; border: 1px solid #222;
                          border-radius: 8px; }
            QLabel#title { color: #ff6b35; font-size: 15px;
                           font-weight: bold; letter-spacing: 1.5px; }
            QPushButton { background: rgba(255,107,53,0.08);
                          border: 1px solid rgba(255,107,53,0.30);
                          border-radius: 6px; color: #ff6b35;
                          padding: 6px 16px; min-height: 28px; }
            QPushButton:hover { background: rgba(255,107,53,0.22);
                                color: #ffffff; }
        )");
        buildUi(title);
    }

protected:
    QVBoxLayout* m_contentLayout;

private:
    void buildUi(const QString& title) {
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(16, 16, 16, 16);
        root->setSpacing(12);

        // Header
        auto* hdr = new QLabel(title, this);
        hdr->setObjectName("title");
        root->addWidget(hdr);

        // Separator
        auto* sep = new QFrame(this);
        sep->setFrameShape(QFrame::HLine);
        sep->setStyleSheet("border: 1px solid #222;");
        root->addWidget(sep);

        // Content card
        auto* card = new QFrame(this);
        card->setObjectName("card");
        m_contentLayout = new QVBoxLayout(card);
        m_contentLayout->setContentsMargins(14, 14, 14, 14);
        root->addWidget(card, 1);

        // Buttons
        auto* btnRow = new QHBoxLayout();
        btnRow->addStretch(1);
        auto* btnClose = new QPushButton("Tutup", this);
        connect(btnClose, &QPushButton::clicked, this, &QDialog::accept);
        btnRow->addWidget(btnClose);
        root->addLayout(btnRow);
    }
};
```"""),

        ("Cara implementasi QGraphicsScene dengan physics force-directed layout di Qt5?",
"""Physics simulation dengan force-directed graph di Qt5:

```cpp
#include <QGraphicsScene>
#include <QTimer>
#include <QVector2D>
#include <cmath>

class PhysicsScene : public QGraphicsScene {
    Q_OBJECT
public:
    explicit PhysicsScene(QObject* parent = nullptr) : QGraphicsScene(parent) {
        setSceneRect(-800, -800, 1600, 1600);
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &PhysicsScene::step);
    }

    void startSim() { m_timer->start(30); }
    void stopSim()  { m_timer->stop(); }

private slots:
    void step() {
        const double K_REP  = 2000.0;  // repulsion constant
        const double K_SPR  = 50.0;    // spring constant
        const double DAMP   = 0.82;    // velocity damping
        const double DT     = 0.35;    // time step
        const double CENTER = 0.025;   // gravity to center

        for (auto* n : m_nodes) {
            QVector2D force(0, 0);
            for (auto* o : m_nodes) {
                if (n == o) continue;
                QVector2D d = n->m_pos - o->m_pos;
                double dist = std::max(d.length(), 0.1);
                if (dist < 300.0)
                    force += d.normalized() * (K_REP / (dist * dist));
            }
            force += -n->m_pos * CENTER;
            n->m_vel = n->m_vel * DAMP + force * DT;
        }

        for (auto* e : m_edges) {
            QVector2D d = e->dst->m_pos - e->src->m_pos;
            double dist  = std::max(d.length(), 0.1);
            double target = (e->src->type == NodeType::Domain) ? 140.0 : 70.0;
            QVector2D att = d.normalized() * K_SPR * (dist - target) * 0.01;
            e->src->m_vel += att;
            e->dst->m_vel -= att;
        }

        for (auto* n : m_nodes) {
            if (n->m_isDragging) {
                n->m_pos = QVector2D(n->pos());
                n->m_vel = {};
            } else {
                n->m_pos += n->m_vel * DT;
                if (n->m_pos.length() > 700.0)
                    n->m_pos = n->m_pos.normalized() * 700.0;
                n->setPos(n->m_pos.toPointF());
            }
        }
        update();
    }

    QTimer* m_timer;
    QList<NodeItem*> m_nodes;
    QList<EdgeItem*> m_edges;
};
```"""),

        ("Cara membuat custom QGraphicsItem yang bisa di-hover dan di-drag?",
"""QGraphicsItem dengan hover glow dan drag support:

```cpp
#include <QGraphicsItem>
#include <QPainter>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneMouseEvent>
#include <QVector2D>
#include <QPainterPath>

class MokoNodeItem : public QGraphicsItem {
public:
    enum class Type { Domain, Bucket, Memory };

    MokoNodeItem(const QString& label, Type t, const QColor& color)
        : m_label(label), m_type(t), m_color(color) {
        setAcceptHoverEvents(true);
        setFlag(QGraphicsItem::ItemIsMovable, true);
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }

    QRectF boundingRect() const override {
        double r = radius() + 6.0;
        return QRectF(-r, -r, 2*r, 2*r);
    }

    void paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) override {
        p->setRenderHint(QPainter::Antialiasing);
        double r = radius();

        // Glow halo saat hover / drag
        if (m_hovered || m_dragging) {
            p->setBrush(QColor(m_color.red(), m_color.green(), m_color.blue(),
                               m_dragging ? 70 : 40));
            p->setPen(Qt::NoPen);
            p->drawEllipse(QRectF(-r-5, -r-5, 2*r+10, 2*r+10));
        }

        // Gradient fill
        QRadialGradient grad(0, 0, r);
        grad.setColorAt(0, (m_type == Type::Domain)
                            ? QColor(10,10,10,220) : m_color);
        grad.setColorAt(1, m_color);
        p->setBrush(grad);

        QPen pen(m_color, (m_type == Type::Domain) ? 2.5 : 1.2);
        if (m_hovered || m_dragging) { pen.setColor(Qt::white); pen.setWidthF(pen.widthF()+1); }
        p->setPen(pen);
        p->drawEllipse(QRectF(-r, -r, 2*r, 2*r));

        // Label
        if (m_type == Type::Domain || m_hovered || m_dragging) {
            p->setPen(QColor(208, 208, 208));
            QFont f = p->font();
            f.setPointSize(8); f.setBold(m_type == Type::Domain);
            p->setFont(f);
            p->drawText(QRectF(-80, r+4, 160, 18), Qt::AlignCenter, m_label);
        }
    }

    QVector2D m_pos, m_vel;
    bool m_dragging = false;

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent*) override { m_hovered=true; update(); }
    void hoverLeaveEvent(QGraphicsSceneHoverEvent*) override { m_hovered=false; update(); }
    void mousePressEvent(QGraphicsSceneMouseEvent* e) override {
        m_dragging=true; m_vel={}; QGraphicsItem::mousePressEvent(e);
    }
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* e) override {
        m_dragging=false; QGraphicsItem::mouseReleaseEvent(e);
    }

private:
    double radius() const {
        return (m_type==Type::Domain) ? 22.0 : (m_type==Type::Bucket) ? 12.0 : 7.0;
    }
    QString m_label;
    Type    m_type;
    QColor  m_color;
    bool    m_hovered = false;
};
```"""),

        ("Implementasikan thread pool di C++ untuk background tasks di MOKO IDE?",
"""Thread pool yang bersih untuk background processing:

```cpp
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <future>
#include <vector>

class MokoThreadPool {
public:
    explicit MokoThreadPool(size_t n = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < n; ++i)
            m_workers.emplace_back([this] { loop(); });
    }

    ~MokoThreadPool() {
        { std::lock_guard<std::mutex> g(m_mtx); m_stop = true; }
        m_cv.notify_all();
        for (auto& t : m_workers) t.join();
    }

    template<typename F, typename... A>
    auto submit(F&& f, A&&... a) -> std::future<std::invoke_result_t<F,A...>> {
        using R = std::invoke_result_t<F, A...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<A>(a)...));
        auto fut = task->get_future();
        { std::lock_guard<std::mutex> g(m_mtx);
          m_queue.emplace([task]{ (*task)(); }); }
        m_cv.notify_one();
        return fut;
    }

private:
    void loop() {
        while (true) {
            std::function<void()> job;
            { std::unique_lock<std::mutex> g(m_mtx);
              m_cv.wait(g, [&]{ return m_stop || !m_queue.empty(); });
              if (m_stop && m_queue.empty()) return;
              job = std::move(m_queue.front()); m_queue.pop(); }
            job();
        }
    }

    std::vector<std::thread>           m_workers;
    std::queue<std::function<void()>>  m_queue;
    std::mutex                         m_mtx;
    std::condition_variable            m_cv;
    bool                               m_stop = false;
};

// Contoh usage di MOKO Stress Test:
// MokoThreadPool pool(4);
// auto fut = pool.submit([]{ return countLinesOfCode("/path/to/project"); });
// int loc = fut.get();
```"""),

        ("Cara membuat Bezier curved edges untuk QGraphicsItem connections?",
"""Bezier curves untuk koneksi seperti benang organik di QGraphicsScene:

```cpp
#include <QGraphicsItem>
#include <QPainter>
#include <QPainterPath>
#include <cmath>

class CurvedEdgeItem : public QGraphicsItem {
public:
    CurvedEdgeItem(QGraphicsItem* src, QGraphicsItem* dst,
                   const QColor& color = Qt::white, bool dashed = false)
        : m_src(src), m_dst(dst), m_color(color), m_dashed(dashed) {
        setZValue(-1);  // Render di bawah node
    }

    QRectF boundingRect() const override {
        if (!m_src || !m_dst) return {};
        return QRectF(m_src->pos(), m_dst->pos()).normalized().adjusted(-30,-30,30,30);
    }

    void paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) override {
        if (!m_src || !m_dst) return;
        p->setRenderHint(QPainter::Antialiasing);

        QPointF p1 = m_src->pos();
        QPointF p2 = m_dst->pos();

        // Hitung control point untuk kurva Bezier
        QPointF mid  = (p1 + p2) / 2.0;
        QPointF diff = p2 - p1;
        double   len  = std::sqrt(diff.x()*diff.x() + diff.y()*diff.y());
        QPointF ctrl  = mid;

        if (len > 0.1) {
            // Normal vector (tegak lurus ke edge), 12% curvature
            double offset = len * 0.12;
            ctrl += QPointF(-diff.y() / len, diff.x() / len) * offset;
        }

        QPainterPath path;
        path.moveTo(p1);
        path.quadTo(ctrl, p2);  // Quadratic Bezier

        QPen pen(m_color, m_highlighted ? 2.5 : (m_dashed ? 1.5 : 1.0));
        if (m_dashed && !m_highlighted) pen.setStyle(Qt::DashLine);
        p->setPen(pen);
        p->drawPath(path);

        // Arrow tip (opsional)
        if (m_showArrow) drawArrow(p, ctrl, p2);
    }

    bool m_highlighted = false;
    bool m_showArrow   = false;

private:
    void drawArrow(QPainter* p, const QPointF& ctrl, const QPointF& tip) {
        QPointF dir = (tip - ctrl).normalized();
        QPointF left  = tip - dir*10 + QPointF(-dir.y(), dir.x())*5;
        QPointF right = tip - dir*10 - QPointF(-dir.y(), dir.x())*5;
        QPolygonF arrow; arrow << tip << left << right;
        p->setBrush(m_color); p->setPen(Qt::NoPen);
        p->drawPolygon(arrow);
    }

    QGraphicsItem* m_src;
    QGraphicsItem* m_dst;
    QColor         m_color;
    bool           m_dashed;
};
```"""),

        ("Cara membuat HTML legend table di QTextBrowser dengan styling Moko v5?",
"""HTML legend table yang clean di QTextBrowser:

```cpp
QTextBrowser* createLegendWidget(QWidget* parent) {
    auto* view = new QTextBrowser(parent);
    view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view->setOpenLinks(false);
    view->setStyleSheet(R"(
        QTextBrowser {
            background: #050505; border: 1px solid #1a1a1a;
            border-radius: 8px; color: #c0c0c0;
            font-family: 'Fira Code', monospace; font-size: 11px;
        }
    )");

    view->setHtml(R"(
<style>
body { margin:0; padding:4px; background:#050505; color:#c0c0c0;
       font-family:'Fira Code',monospace; font-size:10px; }
table { width:100%; border-collapse:collapse; }
th { color:#ff6b35; text-align:left; padding:3px 4px;
     border-bottom:1px solid #2a2a2a; }
td { padding:3px 4px; vertical-align:middle; }
tr:nth-child(even) { background:rgba(255,255,255,0.02); }
.sep { border-top:1px solid #2a2a2a; }
</style>
<table>
<tr><th>Elemen</th><th>Deskripsi</th></tr>
<tr><td><span style="color:#ff6b35;font-size:14px">&#9679;</span> Domain</td>
    <td>Kategori kognitif utama</td></tr>
<tr><td><span style="color:#cc44ff;font-size:14px">&#9679;</span> Bucket</td>
    <td>Indeks SimHash 16-bit</td></tr>
<tr><td><span style="color:#00ff88;font-size:14px">&#9679;</span> Memory</td>
    <td>SHA3 block ingatan riil</td></tr>
<tr class="sep">
    <td style="color:#c0c0c0">&#8213; Solid</td>
    <td>Tautan struktural hirarki</td></tr>
<tr><td style="color:#ff6b35">- - Dash</td>
    <td>RAG highway lintas-domain</td></tr>
<tr><td style="color:#00ff88">&#8213; Green</td>
    <td>Jalur aktif retrieval RAG</td></tr>
</table>
)");
    return view;
}
```"""),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant,
            metadata={"source": "cpp_qt", "type": "qt5_pattern"}))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: ALGORITHMS & DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

def generate_algorithm_samples() -> List[Dict]:
    samples = []

    pairs = [
        ("Implementasikan QuickSort dengan median-of-three pivot di Python.",
"""QuickSort dengan median-of-three untuk performa optimal:

```python
from typing import List

def _median3(arr: List[int], lo: int, hi: int) -> int:
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]: arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:  arr[lo], arr[hi]  = arr[hi],  arr[lo]
    if arr[mid] > arr[hi]: arr[mid], arr[hi] = arr[hi],  arr[mid]
    arr[mid], arr[hi-1] = arr[hi-1], arr[mid]
    return arr[hi-1]

def _partition(arr: List[int], lo: int, hi: int) -> int:
    if hi - lo < 2:
        if arr[lo] > arr[hi]: arr[lo], arr[hi] = arr[hi], arr[lo]
        return lo
    pivot = _median3(arr, lo, hi)
    i, j = lo, hi - 1
    while True:
        i += 1
        while arr[i] < pivot: i += 1
        j -= 1
        while arr[j] > pivot: j -= 1
        if i >= j: break
        arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[hi-1] = arr[hi-1], arr[i]
    return i

def quicksort(arr: List[int], lo: int = 0, hi: int = None) -> None:
    if hi is None: hi = len(arr) - 1
    if lo < hi:
        p = _partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)

# Test
data = [3, 6, 8, 10, 1, 2, 1, 99, 42, 7]
quicksort(data)
print(data)  # [1, 1, 2, 3, 6, 7, 8, 10, 42, 99]
```

Complexity: O(n log n) avg, O(n^2) worst (jarang dgn median-of-3)."""),

        ("Implementasikan LRU Cache O(1) dari scratch di Python.",
"""LRU Cache dengan doubly-linked list + hashmap:

```python
from typing import Optional, TypeVar, Generic

K, V = TypeVar('K'), TypeVar('V')

class _Node(Generic[K, V]):
    __slots__ = ('key','val','prev','next')
    def __init__(self, k=None, v=None):
        self.key, self.val = k, v
        self.prev = self.next = None

class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache: dict = {}
        # Sentinel head/tail
        self.head, self.tail = _Node(), _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: K) -> Optional[V]:
        if key not in self.cache: return None
        n = self.cache[key]
        self._move_front(n)
        return n.val

    def put(self, key: K, val: V) -> None:
        if key in self.cache:
            n = self.cache[key]; n.val = val; self._move_front(n)
        else:
            if len(self.cache) >= self.cap:
                lru = self.tail.prev
                self._remove(lru); del self.cache[lru.key]
            n = _Node(key, val)
            self.cache[key] = n; self._add_front(n)

    def _add_front(self, n):
        n.prev = self.head; n.next = self.head.next
        self.head.next.prev = n; self.head.next = n

    def _remove(self, n):
        n.prev.next = n.next; n.next.prev = n.prev

    def _move_front(self, n):
        self._remove(n); self._add_front(n)

# Usage di MOKO RAG Cache:
# cache = LRUCache[str, list](capacity=1000)
# cache.put("binary search python", search_results)
# result = cache.get("binary search python")  # O(1)
```"""),

        ("Implementasikan Trie untuk autocomplete di MOKO IDE command palette.",
"""Trie dengan frekuensi ranking untuk command autocomplete:

```python
from typing import List, Optional, Dict

class TrieNode:
    __slots__ = ('ch','is_end','hits','data')
    def __init__(self):
        self.ch: Dict[str,'TrieNode'] = {}
        self.is_end = False
        self.hits   = 0
        self.data   = None

class CommandTrie:
    def __init__(self): self.root = TrieNode()

    def insert(self, cmd: str, data: dict = None) -> None:
        node = self.root
        for c in cmd.lower():
            node.ch.setdefault(c, TrieNode()); node = node.ch[c]
        node.is_end = True; node.data = data or {}

    def search(self, prefix: str, top: int = 10) -> List[dict]:
        node = self.root
        for c in prefix.lower():
            if c not in node.ch: return []
            node = node.ch[c]
        results = []
        self._dfs(node, prefix.lower(), results)
        results.sort(key=lambda x: x.get('hits',0), reverse=True)
        return results[:top]

    def _dfs(self, n: TrieNode, cur: str, out: list):
        if n.is_end: out.append({'cmd': cur, 'hits': n.hits, **n.data})
        for c, child in n.ch.items(): self._dfs(child, cur+c, out)

    def hit(self, cmd: str) -> None:
        node = self.root
        for c in cmd.lower():
            if c not in node.ch: return
            node = node.ch[c]
        if node.is_end: node.hits += 1

# Setup MOKO commands:
trie = CommandTrie()
trie.insert("/graphify", {"desc": "Memory Monitor",    "icon": "net"})
trie.insert("/stress",   {"desc": "Cognitive Stress",  "icon": "flash"})
trie.insert("/model",    {"desc": "Model Manager",     "icon": "robot"})
trie.insert("/clear",    {"desc": "Clear History",     "icon": "trash"})
print(trie.search("/g"))   # [{'cmd': '/graphify', ...}]
```"""),

        ("Implementasikan Dijkstra shortest path di Python untuk domain graph MOKO.",
"""Dijkstra dengan heapq untuk MOKO domain routing:

```python
import heapq
from typing import Dict, List, Tuple, Optional

def dijkstra(
    graph: Dict[str, List[Tuple[str, float]]],
    start: str,
    end: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    dist = {n: float('inf') for n in graph}
    dist[start] = 0.0
    prev: Dict[str, Optional[str]] = {n: None for n in graph}
    heap = [(0.0, start)]
    vis: set = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if end and u == end: break
        for v, w in graph.get(u, []):
            if v not in vis and d + w < dist[v]:
                dist[v] = d + w; prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, prev

def path(prev, start, end) -> List[str]:
    p, cur = [], end
    while cur: p.append(cur); cur = prev[cur]
    p.reverse()
    return p if p and p[0] == start else []

# Graf cross-domain MOKO RAG:
moko_graph = {
    "code":     [("math",0.3),("security",0.5),("general",0.7)],
    "math":     [("code",0.3),("science",0.4)],
    "security": [("code",0.5),("general",0.4)],
    "science":  [("math",0.4),("general",0.3)],
    "general":  [("code",0.7),("math",0.6),("security",0.4)],
}

d, p = dijkstra(moko_graph, "security", "math")
print(" -> ".join(path(p, "security", "math")), f"cost={d['math']:.2f}")
# security -> general -> math, cost=1.00
```"""),

        ("Implementasikan SimHash untuk near-duplicate detection di MOKO RAG.",
"""SimHash 64-bit untuk mendeteksi near-duplicate dokumen:

```python
import hashlib
import re
from typing import List

class SimHash:
    BITS = 64

    def __init__(self, text: str):
        self.hash = self._compute(text)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\\w+\b', text)
        # Bigrams untuk context awareness
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams

    def _compute(self, text: str) -> int:
        tokens = self._tokenize(text)
        v = [0] * self.BITS
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(self.BITS):
                v[i] += 1 if (h >> i) & 1 else -1
        return sum(1 << i for i in range(self.BITS) if v[i] > 0)

    def hamming(self, other: 'SimHash') -> int:
        # Hamming distance: jumlah bit berbeda
        xor = self.hash ^ other.hash
        return bin(xor).count('1')

    def is_near_duplicate(self, other: 'SimHash', threshold: int = 3) -> bool:
        # threshold=3: toleransi 3 bit berbeda (sekitar 95% mirip)
        return self.hamming(other) <= threshold

# Usage di MOKO Omni dedup:
doc1 = SimHash("QuickSort adalah algoritma sorting efisien berbasis divide and conquer")
doc2 = SimHash("QuickSort ialah algoritma sort yang efisien menggunakan divide and conquer")
doc3 = SimHash("Fibonacci sequence adalah barisan angka di mana setiap angka adalah jumlah dua angka sebelumnya")

print(doc1.hamming(doc2))           # ~2-4 (near-dup)
print(doc1.is_near_duplicate(doc2)) # True
print(doc1.is_near_duplicate(doc3)) # False
```"""),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant,
            metadata={"source": "algorithms", "type": "algo_ds"}))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: SECURITY & ETHICAL HACKING
# ─────────────────────────────────────────────────────────────────────────────

def generate_security_samples() -> List[Dict]:
    samples = []

    pairs = [
        ("Bagaimana cara melakukan port scanning dengan Python untuk network recon?",
"""Port scanner paralel untuk network reconnaissance (ethical use only):

```python
import socket, threading, queue
from typing import List, Tuple

class PortScanner:
    SERVICES = {
        21:'FTP',22:'SSH',23:'Telnet',25:'SMTP',53:'DNS',
        80:'HTTP',443:'HTTPS',445:'SMB',3306:'MySQL',
        3389:'RDP',5432:'PG',6379:'Redis',8080:'HTTP-Alt',
    }

    def __init__(self, host: str, timeout: float = 0.8, threads: int = 100):
        self.host, self.timeout, self.threads = host, timeout, threads
        self.open: List[Tuple[int,str]] = []
        self._lock = threading.Lock()

    def _check(self, port: int) -> bool:
        try:
            s = socket.socket(); s.settimeout(self.timeout)
            ok = s.connect_ex((self.host, port)) == 0
            s.close(); return ok
        except: return False

    def _worker(self, q: queue.Queue):
        while True:
            try: port = q.get_nowait()
            except queue.Empty: break
            if self._check(port):
                svc = self.SERVICES.get(port,'unknown')
                with self._lock: self.open.append((port, svc))
            q.task_done()

    def scan(self, start: int = 1, end: int = 1024) -> List[Tuple[int,str]]:
        q: queue.Queue = queue.Queue()
        for p in range(start, end+1): q.put(p)
        ts = [threading.Thread(target=self._worker, args=(q,), daemon=True)
              for _ in range(min(self.threads, end-start+1))]
        for t in ts: t.start()
        q.join()
        self.open.sort()
        return self.open

# PERINGATAN: Hanya gunakan pada sistem yang Anda miliki atau punya izin!
# s = PortScanner("192.168.1.1", timeout=0.5)
# for port, svc in s.scan(1, 1024):
#     print(f"  {port:5d}/tcp  OPEN  {svc}")
```"""),

        ("Implementasikan Caesar cipher dan Vigenere cipher di Python.",
"""Caesar dan Vigenere cipher untuk CTF dan kriptografi dasar:

```python
import string

class CaesarCipher:
    def __init__(self, shift: int):
        self.shift = shift % 26

    def encrypt(self, text: str) -> str:
        result = []
        for ch in text:
            if ch.isupper():
                result.append(chr((ord(ch) - 65 + self.shift) % 26 + 65))
            elif ch.islower():
                result.append(chr((ord(ch) - 97 + self.shift) % 26 + 97))
            else:
                result.append(ch)
        return ''.join(result)

    def decrypt(self, text: str) -> str:
        return CaesarCipher(-self.shift).encrypt(text)

    @staticmethod
    def brute_force(cipher: str) -> dict:
        # Coba semua 26 kemungkinan shift
        return {s: CaesarCipher(s).decrypt(cipher) for s in range(26)}


class VigenereCipher:
    def __init__(self, key: str):
        self.key = key.upper()

    def _apply(self, text: str, encrypt: bool) -> str:
        result, ki = [], 0
        for ch in text:
            if ch.isalpha():
                shift = ord(self.key[ki % len(self.key)]) - 65
                if not encrypt: shift = -shift
                base = 65 if ch.isupper() else 97
                result.append(chr((ord(ch) - base + shift) % 26 + base))
                ki += 1
            else:
                result.append(ch)
        return ''.join(result)

    def encrypt(self, text: str) -> str: return self._apply(text, True)
    def decrypt(self, text: str) -> str: return self._apply(text, False)

# Test:
c = CaesarCipher(13)  # ROT13
print(c.encrypt("MOKO IDE"))   # "ZBXB VQR"
print(c.decrypt("ZBXB VQR"))   # "MOKO IDE"

v = VigenereCipher("MOKO")
enc = v.encrypt("Hello World")
print(enc)               # "TWZBY IFLZP"
print(v.decrypt(enc))    # "HELLO WORLD"
```"""),

        ("Cara mengimplementasikan XOR-based simple encryption untuk data obfuscation?",
"""XOR encryption dengan key scheduling untuk data protection:

```python
import os
import hashlib
from typing import Union

class XorCipher:
    # XOR encryption dengan key stretching menggunakan SHA-256

    def __init__(self, password: str):
        # Derive 256-byte keystream dari password
        self.key = self._derive_key(password, length=256)

    def _derive_key(self, password: str, length: int) -> bytes:
        key = b''
        i   = 0
        while len(key) < length:
            chunk = hashlib.sha256(password.encode() + i.to_bytes(4,'big')).digest()
            key  += chunk
            i    += 1
        return key[:length]

    def crypt(self, data: Union[bytes, str]) -> bytes:
        # Encrypt atau decrypt (XOR bersifat involutory)
        if isinstance(data, str): data = data.encode('utf-8')
        key_len = len(self.key)
        return bytes(b ^ self.key[i % key_len] for i, b in enumerate(data))

    def encrypt_file(self, src: str, dst: str) -> None:
        with open(src, 'rb') as f: plain = f.read()
        with open(dst, 'wb') as f: f.write(self.crypt(plain))

    def decrypt_file(self, src: str, dst: str) -> None:
        self.encrypt_file(src, dst)  # Same operation


# Usage:
xor = XorCipher("moko_secret_key_2026")
enc = xor.crypt("sensitive data")
dec = xor.crypt(enc)
print(dec.decode())   # "sensitive data"

# File encryption:
# xor.encrypt_file("config.json", "config.enc")
# xor.decrypt_file("config.enc", "config_restored.json")
```

**Catatan:** Untuk production security, gunakan AES-256-GCM dari library cryptography. XOR hanya untuk obfuscation/CTF."""),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant, system=SYSTEM_SECURITY,
            metadata={"source": "security", "type": "ethical_hacking"}))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: MOKO API INTEGRATION EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def generate_integration_samples() -> List[Dict]:
    samples = []

    pairs = [
        ("Cara menggunakan MOKO Omni RAG untuk menyimpan dan mencari knowledge?",
"""Cara pakai MOKO Omni RAG system dari Python:

```python
from pathlib import Path
# Import MOKO storage modules
# (pastikan PROJECT_DIR ada di PYTHONPATH)
import sys
sys.path.insert(0, str(Path("/home/user/Documents/Linux/MOKO_OS_Project")))

from moko_core.moko_memory.omni_storage import OmniStorage

PROJECT_DIR = Path("/home/user/Documents/Linux/MOKO_OS_Project")
storage = OmniStorage(omni_dir=PROJECT_DIR / ".moko_omni")

# ── Simpan knowledge ke domain 'code' ─────────────────────────────
entry = {
    "content": "QuickSort adalah algoritma sorting O(n log n) rata-rata.",
    "source":  "algorithm_docs",
    "domain":  "code",
    "tags":    ["sorting", "algorithm"],
}
entry_id = storage.store(entry)
print(f"Stored: {entry_id}")

# ── Semantic Search ────────────────────────────────────────────────
results = storage.search(
    query   = "algoritma sorting paling cepat",
    top_k   = 5,
    domain  = "code",   # None = semua domain (lintas-domain RAG)
    min_score = 0.65
)

for r in results:
    print(f"[{r['score']:.3f}] {r['content'][:80]}...")
    print(f"  Source: {r.get('source')} | Domain: {r.get('domain')}")

# ── Cross-domain search (core MOKO RAG feature) ────────────────────
cross_results = storage.search(
    query  = "neural network optimization math",
    top_k  = 10,
    domain = None,  # Semua domain: code, math, science, dll
)
print(f"Cross-domain: {len(cross_results)} results from multiple domains")
```"""),

        ("Bagaimana cara MOKO IDE (C++) berkomunikasi dengan Python backend?",
"""Komunikasi MOKO IDE C++ dengan Python inference backend via HTTP:

```cpp
// helper_engine.cpp — Async HTTP request ke llama.cpp server
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

void HelperEngine::askAsync(
    const QString& prompt,
    std::function<void(const QString&)> onChunk,
    std::function<void()> onDone)
{
    const int PORT = 11434;  // llama-server default port

    QJsonObject payload;
    payload["model"]       = "moko-coder";
    payload["temperature"] = 0.1;
    payload["stream"]      = true;   // Enable SSE streaming

    QJsonArray msgs;
    QJsonObject msg; msg["role"] = "user"; msg["content"] = prompt;
    msgs.append(msg); payload["messages"] = msgs;

    QNetworkRequest req(QUrl(
        QString("http://127.0.0.1:%1/v1/chat/completions").arg(PORT)));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->post(req, QJsonDocument(payload).toJson());

    connect(reply, &QNetworkReply::readyRead, [reply, onChunk]() {
        // Parse SSE: "data: {json}\n\n"
        QByteArray raw = reply->readAll();
        for (auto& line : raw.split('\n')) {
            if (!line.startsWith("data: ")) continue;
            QByteArray json = line.mid(6).trimmed();
            if (json == "[DONE]") continue;
            QJsonDocument doc = QJsonDocument::fromJson(json);
            QString chunk = doc.object()["choices"]
                              .toArray()[0].toObject()
                              ["delta"].toObject()
                              ["content"].toString();
            if (!chunk.isEmpty()) onChunk(chunk);
        }
    });

    connect(reply, &QNetworkReply::finished, [reply, onDone]() {
        onDone(); reply->deleteLater();
    });
}
```"""),

        ("Cara membuat dan mendaftarkan domain baru di MOKO RAG system?",
"""Cara tambah domain baru ke MOKO Omni knowledge base:

```python
from pathlib import Path
from moko_core.moko_memory.omni_storage import OmniStorage

PROJECT_DIR = Path("/home/user/Documents/Linux/MOKO_OS_Project")
OMNI_DIR    = PROJECT_DIR / ".moko_omni"

# ── Step 1: Buat direktori domain ─────────────────────────────────
new_domain = "qt_cpp"
(OMNI_DIR / new_domain).mkdir(parents=True, exist_ok=True)

# ── Step 2: Inisialisasi storage ──────────────────────────────────
storage = OmniStorage(omni_dir=OMNI_DIR)

# ── Step 3: Batch index dokumen ───────────────────────────────────
qt_docs = [
    ("QWidget adalah base class semua UI objects di Qt5. "
     "Parent-child relationship mengatur lifecycle dan layout otomatis.",
     "qt5_docs", ["qwidget", "ui"]),
    ("QGraphicsScene menyediakan 2D canvas untuk QGraphicsItem. "
     "Physics simulation bisa diimplementasikan lewat QTimer + force computation.",
     "qt5_docs", ["graphics", "simulation"]),
    ("QTextBrowser mendukung HTML rendering dengan CSS inline. "
     "Gunakan setHtml() untuk menampilkan tabel dan formatted text.",
     "qt5_docs", ["textbrowser", "html"]),
]

for content, source, tags in qt_docs:
    entry_id = storage.store({
        "content": content, "source": source,
        "domain": new_domain, "tags": tags,
    })
    print(f"[OK] Indexed: {entry_id}")

# ── Step 4: Verifikasi ─────────────────────────────────────────────
results = storage.search("QWidget parent layout", domain="qt_cpp", top_k=3)
print(f"Verification: {len(results)} results found in domain 'qt_cpp'")
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
```"""),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant,
            metadata={"source": "moko_integration", "type": "api_usage"}))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: MULTI-TURN DEBUGGING CONVERSATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_multiturn_samples() -> List[Dict]:
    samples = []

    conversations = [
        [
            "Saya punya error: `KeyError: 'messages'` saat load JSONL dataset.",
            """Error ini karena ada line JSONL yang formatnya berbeda (bukan ChatML).
Cek dulu dengan script ini:

```python
import json

def validate_jsonl(path: str):
    valid, errors = 0, []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                d = json.loads(line)
                if 'messages' not in d:
                    errors.append((i, "Missing 'messages'", str(d)[:60]))
                else:
                    valid += 1
            except Exception as e:
                errors.append((i, str(e), line[:60]))
    print(f"Valid: {valid}, Errors: {len(errors)}")
    for ln, err, sample in errors[:5]:
        print(f"  Line {ln}: {err} -> {sample}")

validate_jsonl("dataset.jsonl")
```
Jalankan ini dan bagikan outputnya.""",

            "Output: Valid: 2770, Errors: 6. Line 1234: Missing 'messages' -> `{'instruction': 'buat sorting', 'output': '...'}`",
            """Tepat! Ada campuran format Alpaca (`instruction/output`) dengan ChatML (`messages`). Fix otomatis:

```python
import json

def fix_mixed_jsonl(src: str, dst: str, system: str = ""):
    fixed = 0
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if 'messages' not in d:
                # Convert Alpaca -> ChatML
                msgs = []
                if system: msgs.append({"role":"system","content":system})
                user_txt = d.get('instruction','')
                if d.get('input'): user_txt += f"\n\nInput:\n{d['input']}"
                msgs.append({"role":"user",      "content": user_txt})
                msgs.append({"role":"assistant", "content": d.get('output','')})
                d = {"messages": msgs}
                fixed += 1
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"Fixed {fixed} records")

fix_mixed_jsonl("dataset.jsonl", "dataset_fixed.jsonl",
                system="You are MOKO Coder...")
```
Setelah itu validasi ulang file_fixed.jsonl.""",
        ],

        [
            "Qt compile error: `error: no matching function for call to 'QVector2D::QVector2D(QPointF)'`",
            """Qt tidak punya constructor `QVector2D(QPointF)` secara langsung. Fix:

```cpp
// SALAH:
QVector2D v = QVector2D(somePoint);

// BENAR (pakai x,y eksplisit):
QVector2D v(somePoint.x(), somePoint.y());

// Atau buat helper:
static QVector2D toV2(const QPointF& p) {
    return QVector2D(float(p.x()), float(p.y()));
}
```
Tunjukkan baris kode yang bermasalah.""",

            "Kodenya: `n->m_pos = QVector2D(n->pos());` di dalam runSimulationStep",
            """`n->pos()` mengembalikan `QPointF`. Fix:

```cpp
// SALAH:
n->m_pos = QVector2D(n->pos());

// BENAR:
n->m_pos = QVector2D(float(n->pos().x()), float(n->pos().y()));

// Atau jika sudah ada helper toV2():
n->m_pos = toV2(n->pos());
```

Cek semua tempat serupa:
```bash
grep -n "QVector2D(.*->pos())" moko_ide_cpp/*.cpp
```""",
        ],

        [
            "Python error saat training: `CUDA out of memory` padahal batch_size sudah 1.",
            """CUDA OOM dengan batch_size=1 biasanya dari `max_seq_length` terlalu besar atau gradient accumulation. Coba:

```python
# Di training_config, turunkan max_seq_length:
TRAINING_CONFIG = {
    "max_seq_length": 512,      # Turun dari 1536/2048 -> 512 dulu
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    # ...
}
```

Atau paksa garbage collect sebelum training:
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
```

Berapa total VRAM GPU Anda?""",

            "RTX 2050 4GB. Error muncul di epoch pertama step 50.",
            """4GB sangat terbatas untuk training. Gunakan konfigurasi ini yang optimal untuk RTX 2050:

```python
# Config khusus RTX 2050 4GB:
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Double quantization hemat ~0.4 GB
)

TRAINING_CONFIG = {
    "max_seq_length": 512,           # MAX untuk 4GB + NF4
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, # Effective batch = 8
    "gradient_checkpointing": True,   # Hemat VRAM ~40%
    "optim": "paged_adamw_8bit",      # Optimizer 8-bit
    "fp16": False,
    "bf16": True,
}

# Di SFTTrainer tambahkan:
# max_seq_length=512, packing=False
```

Dengan konfigurasi ini harusnya fit di 3.5-4GB VRAM.""",
        ],
    ]

    for turns in conversations:
        msgs = [{"role": "system", "content": SYSTEM_CODER}]
        roles = (["user","assistant"] * 10)[:len(turns)]
        for role, content in zip(roles, turns):
            msgs.append({"role": role, "content": content})
        samples.append({
            "messages": msgs,
            "metadata": {"source": "multiturn", "type": "debugging"}
        })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6: CHAIN-OF-THOUGHT REASONING
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning_samples() -> List[Dict]:
    samples = []

    pairs = [
        ("Saya ingin membuat semantic cache untuk MOKO RAG. Apa pendekatan terbaik?",
"""Saya analisis problem ini secara sistematis.

**Step 1 — Identifikasi Bottleneck RAG**
Latency RAG berasal dari:
1. Embedding query (~50-100ms, model inference)
2. Vector similarity search (~10-50ms)
3. Context injection

Bottleneck terbesar: #1 (embedding). Jika query mirip berulang, embedding bisa di-cache.

**Step 2 — Jenis Cache yang Cocok**
- Exact cache (md5 key): miss rate tinggi — "sorting array" vs "urutkan array" = 2 key berbeda
- **Semantic cache** (cosine similarity): cache berdasarkan kedekatan makna — lebih powerful

**Step 3 — Implementasi Semantic LRU Cache**

```python
import numpy as np
import time
from typing import Optional, List

class SemanticRAGCache:
    def __init__(self, embed_fn, capacity: int = 500, threshold: float = 0.92):
        self.embed     = embed_fn      # fungsi embedding -> np.array
        self.capacity  = capacity
        self.threshold = threshold     # >= 0.92 = near-identical query
        self._store    = []            # [(emb, query, results, ts, hits)]
        self._hits = self._misses = 0

    def get(self, query: str) -> Optional[List[dict]]:
        q_emb = self.embed(query)
        best_score, best_result = 0.0, None
        for emb, _, results, ts, hits in self._store:
            # Cosine similarity
            score = float(np.dot(q_emb, emb) /
                          (np.linalg.norm(q_emb)*np.linalg.norm(emb) + 1e-9))
            if score > best_score:
                best_score, best_result = score, results
        if best_score >= self.threshold:
            self._hits += 1; return best_result
        self._misses += 1; return None

    def put(self, query: str, results: List[dict]) -> None:
        emb = self.embed(query)
        if len(self._store) >= self.capacity:
            self._store.pop(0)   # Evict oldest (simple FIFO, bisa diganti LRU)
        self._store.append((emb, query, results, time.time(), 0))

    @property
    def hit_rate(self) -> float:
        t = self._hits + self._misses
        return self._hits / t if t else 0.0
```

**Step 4 — Expected Impact**
- Hit rate: 40-60% pada pola query normal
- Memory: 500 entries x 1536 floats x 4 bytes = ~3MB — aman
- Latency saved: 50-100ms per cache hit

**Kesimpulan:** Gunakan threshold 0.92 (terlalu rendah -> false positive, terlalu tinggi -> miss rate tinggi). Monitor hit_rate dan tuning sesuai kebutuhan."""),

        ("Bagaimana cara merancang sistem fine-tuning yang optimal untuk model 7B di GPU 4GB?",
"""Analisis sistematis untuk fine-tuning 7B di 4GB VRAM.

**Step 1 — Kalkulasi Memory**
Model 7B BF16 = 14GB. Tidak muat di 4GB. Solusi: QLoRA.

QLoRA memory breakdown:
- Model weights (INT4/NF4): 7B x 0.5 bytes = ~3.5GB
- LoRA adapters (BF16): r=64 x layers x params ~200MB
- Activations + gradients: ~0.5-1GB (tergantung seq_len)
- Optimizer state (8-bit Adam): ~400MB
- Total: ~4.1-4.6GB → **sangat mepet untuk 4GB**

**Step 2 — Teknik Hemat VRAM**
1. `load_in_4bit=True` + `bnb_4bit_use_double_quant=True` → -0.4GB
2. `gradient_checkpointing=True` → -40% activation memory
3. `max_seq_length=512` (bukan 2048) → -4x memory quadratic
4. `paged_adamw_8bit` → optimizer di CPU memory jika perlu

**Step 3 — Konfigurasi Rekomendasi**

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit             = True,
    bnb_4bit_quant_type      = "nf4",
    bnb_4bit_compute_dtype   = torch.bfloat16,
    bnb_4bit_use_double_quant= True,   # Double quant hemat ~0.4GB
)

lora_config = {
    "r"          : 32,        # Lebih kecil dari 64 untuk hemat VRAM
    "lora_alpha" : 64,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                       "gate_proj","up_proj","down_proj"],
    "lora_dropout": 0.05,
}

training = {
    "max_seq_length"              : 512,
    "per_device_train_batch_size" : 1,
    "gradient_accumulation_steps" : 16,  # Effective batch = 16
    "gradient_checkpointing"      : True,
    "optim"                       : "paged_adamw_8bit",
    "learning_rate"               : 1e-4,
    "num_train_epochs"            : 3,
    "bf16"                        : True,
}
```

**Step 4 — Monitoring VRAM**
```python
# Pantau selama training:
import torch
def vram_status():
    alloc = torch.cuda.memory_allocated()/1e9
    total = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f"VRAM: {alloc:.2f}/{total:.2f} GB ({alloc/total*100:.0f}%)")
```

**Kesimpulan:** Dengan konfigurasi ini, 7B QLoRA harusnya bisa fit di 3.8-4.0GB. Jika masih OOM, turunkan r=16 atau gunakan gradient_checkpointing lebih agresif."""),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant,
            metadata={"source": "reasoning", "type": "chain_of_thought"}))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7: DOCS → Q&A
# ─────────────────────────────────────────────────────────────────────────────

def generate_docs_samples() -> List[Dict]:
    samples = []
    docs_dir = PROJECT_DIR / "docs" / "riset"
    if not docs_dir.exists():
        log(f"Docs dir not found: {docs_dir}", "WARN")
        return samples

    header_re = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)

    for md_file in sorted(docs_dir.glob("*.md")):
        try:
            content  = md_file.read_text(encoding='utf-8', errors='ignore')
            headers  = header_re.findall(content)
            parts    = header_re.split(content)

            for i, header in enumerate(headers):
                if i + 1 >= len(parts): continue
                body = parts[i + 1].strip()
                if len(body) < 80: continue

                user = random.choice([
                    f"Jelaskan '{header}' dalam konteks MOKO OS.",
                    f"Apa yang dimaksud dengan {header} di MOKO?",
                    f"Bagaimana {header} bekerja di MOKO OS?",
                ])
                assistant = f"**{header}**\n\n{body[:1000]}"
                samples.append(make_sample(user, assistant,
                    metadata={"source": f"docs/{md_file.name}", "type": "documentation"}))
        except Exception as e:
            log(f"Error {md_file.name}: {e}", "WARN")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8: PROGRAMMING & COMMON SYNTAX/RUNTIME ERRORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_programming_error_samples() -> List[Dict]:
    samples = []

    pairs = [
        # Pair 1: Segmentation Fault in C++
        (
            "Saya mendapat error Segmentation Fault (core dumped) di C++. Bagaimana cara mencari penyebab dan memperbaikinya?",
            "Segmentation Fault (SIGSEGV) terjadi ketika program mencoba mengakses area memori yang tidak diizinkan atau tidak didefinisikan.\n\n"
            "**Penyebab Umum:**\n"
            "1. **Null pointer dereference**: Mengakses atau menulis ke pointer bernilai `nullptr`.\n"
            "2. **Out of bounds access**: Mengakses elemen di luar batas indeks `std::vector` atau array C-style.\n"
            "3. **Stack overflow**: Fungsi rekursif tanpa basis kasus yang menguras memori stack.\n"
            "4. **Use after free**: Mengakses pointer setelah memori didealloc menggunakan `delete` atau `free`.\n\n"
            "**Cara Debugging dengan GDB:**\n"
            "```bash\n"
            "g++ -g main.cpp -o program  # Wajib compile dengan flag -g\n"
            "gdb ./program\n"
            "(gdb) run\n"
            "# Program akan crash dan menunjukkan baris error\n"
            "(gdb) backtrace  # Menunjukkan urutan pemanggilan fungsi (call stack)\n"
            "```\n\n"
            "**Contoh & Solusi (Null Pointer):**\n"
            "```cpp\n"
            "// ❌ SALAH: Langsung mengakses isi pointer tanpa check\n"
            "void printLength(std::string* str) {\n"
            "    std::cout << str->length() << std::endl; // Crash jika str == nullptr\n"
            "}\n"
            "\n"
            "// ✅ BENAR: Selalu validasi pointer terlebih dahulu\n"
            "void printLength(std::string* str) {\n"
            "    if (!str) {\n"
            "        std::cerr << \"Error: String pointer is null!\" << std::endl;\n"
            "        return;\n"
            "    }\n"
            "    std::cout << str->length() << std::endl;\n"
            "}\n"
            "```"
        ),

        # Pair 2: Python UnboundLocalError
        (
            "Mengapa saya mendapatkan error 'UnboundLocalError: local variable referenced before assignment' di Python?",
            "Error ini terjadi karena Anda mencoba mengakses variabel lokal di dalam scope fungsi sebelum variabel tersebut dideklarasikan/diisi nilai, padahal variabel dengan nama yang sama ada di scope luar (global).\n\n"
            "Python melihat ada assignment terhadap nama variabel tersebut di dalam fungsi, sehingga secara otomatis menganggap variabel tersebut adalah **lokal** untuk seluruh fungsi, menyembunyikan variabel luar.\n\n"
            "**Contoh Kasus & Solusi:**\n"
            "```python\n"
            "count = 10\n"
            "\n"
            "# ❌ SALAH: Memicu UnboundLocalError\n"
            "def increment():\n"
            "    count = count + 1  # Python menganggap count lokal karena ada 'count = ...'\n"
            "    print(count)\n"
            "\n"
            "# ✅ SOLUSI 1: Gunakan keyword 'global' jika ingin memodifikasi variabel luar\n"
            "def increment_global():\n"
            "    global count\n"
            "    count += 1\n"
            "    print(count)\n"
            "\n"
            "# ✅ SOLUSI 2: Lewatkan sebagai parameter (Pendekatan Pure Function - Direkomendasikan)\n"
            "def increment_pure(current_count):\n"
            "    return current_count + 1\n"
            "```"
        ),

        # Pair 3: Python Mutable Default Arguments
        (
            "Mengapa nilai default list di fungsi Python menyimpan data dari pemanggilan sebelumnya? Contoh: `def add_item(val, items=[])`.",
            "Di Python, nilai parameter default dievaluasi **hanya sekali** ketika modul dimuat (compile time), bukan setiap kali fungsi dipanggil.\n\n"
            "Jika Anda menggunakan objek yang bersifat *mutable* (dapat dirubah nilainya seperti `list`, `dict`, atau `set`) sebagai nilai default, objek yang sama akan digunakan kembali pada semua pemanggilan fungsi berikutnya.\n\n"
            "**Contoh Bug & Solusinya:**\n"
            "```python\n"
            "# ❌ SALAH: items=[] akan terus terakumulasi datanya\n"
            "def append_to(element, items=[]):\n"
            "    items.append(element)\n"
            "    return items\n"
            "\n"
            "print(append_to(1))  # [1]\n"
            "print(append_to(2))  # [1, 2] -- Bug! Seharusnya [2]\n"
            "\n"
            "# ✅ BENAR: Gunakan None sebagai sentinel value\n"
            "def append_to_safe(element, items=None):\n"
            "    if items is None:\n"
            "        items = []  # Buat list baru di setiap pemanggilan fungsi\n"
            "    items.append(element)\n"
            "    return items\n"
            "\n"
            "print(append_to_safe(1))  # [1]\n"
            "print(append_to_safe(2))  # [2] -- Benar!\n"
            "```"
        ),

        # Pair 4: Rust Borrow Checker error
        (
            "Bagaimana cara mengatasi error Rust borrow checker: 'cannot borrow `x` as mutable because it is also borrowed as immutable'?",
            "Aturan peminjaman (borrowing rules) di Rust menyatakan:\n"
            "1. Anda dapat memiliki banyak immutable references (`&T`) sekaligus ke suatu resource.\n"
            "2. Anda **hanya** dapat memiliki **satu** mutable reference (`&mut T`) dalam satu waktu.\n"
            "3. Anda tidak boleh memiliki mutable reference ketika immutable reference sedang aktif.\n\n"
            "Ini menjamin keamanan thread dan mencegah data race.\n\n"
            "**Contoh Kasus & Solusi:**\n"
            "```rust\n"
            "fn main() {\n"
            "    let mut x = vec![1, 2, 3];\n"
            "\n"
            "    // ❌ SALAH: Immutable borrow aktif bersamaan dengan mutable borrow\n"
            "    let r1 = &x;     // Immutable borrow dimulai\n"
            "    let r2 = &mut x; // ❌ ERROR: cannot borrow as mutable!\n"
            "    println!(\"{:?}\", r1); // r1 digunakan di sini, menjaga borrow aktif\n"
            "\n"
            "    // ✅ SOLUSI 1: Batasi scope borrow dengan curly braces\n"
            "    {\n"
            "        let r1 = &x; \n"
            "        println!(\"{:?}\", r1); // r1 mati di akhir scope\n"
            "    }\n"
            "    let r2 = &mut x; // Aman: r1 sudah tidak aktif\n"
            "    r2.push(4);\n"
            "\n"
            "    // ✅ SOLUSI 2: Manfaatkan Non-Lexical Lifetimes (NLL)\n"
            "    let r1 = &x;\n"
            "    println!(\"{:?}\", r1); // Penggunaan terakhir r1\n"
            "    // Rust tahu r1 tidak lagi dipakai di bawah baris ini\n"
            "    let r2 = &mut x;      // Aman!\n"
            "    r2.push(4);\n"
            "}\n"
            "```"
        ),

        # Pair 5: C++ Memory Leak
        (
            "Bagaimana cara mendeteksi memory leak di C++ dan bagaimana cara mengatasinya menggunakan RAII / Smart Pointers?",
            "Memory leak terjadi ketika Anda mengalokasikan memori dinamis di heap menggunakan operator `new` (atau `malloc`), tetapi lupa melepaskannya kembali menggunakan `delete` (atau `free`) sebelum program kehilangan pointer ke memori tersebut.\n\n"
            "**Mendeteksi Memory Leak:**\n"
            "Gunakan tools dynamic analysis seperti **Valgrind**:\n"
            "```bash\n"
            "valgrind --leak-check=full ./program\n"
            "```\n\n"
            "**Solusi: Resource Acquisition Is Initialization (RAII) & Smart Pointers:**\n"
            "Hindari alokasi manual `new` dan `delete`. Gunakan `std::unique_ptr` atau `std::shared_ptr` dari header `<memory>`.\n\n"
            "```cpp\n"
            "#include <memory>\n"
            "\n"
            "// ❌ SALAH: Menggunakan raw pointer rawan memory leak\n"
            "void rawPointerLeak() {\n"
            "    int* raw = new int(42);\n"
            "    if (someCondition()) {\n"
            "        return; // Memory leak! Pointer 'raw' tidak di-delete\n"
            "    }\n"
            "    delete raw;\n"
            "}\n"
            "\n"
            "// ✅ BENAR: Gunakan std::unique_ptr\n"
            "void smartPointerSafe() {\n"
            "    std::unique_ptr<int> smart = std::make_unique<int>(42);\n"
            "    if (someCondition()) {\n"
            "        return; // Aman! Memori otomatis dibebaskan ketika scope berakhir\n"
            "    }\n"
            "    // Memori dilepas secara otomatis\n"
            "}\n"
            "```"
        ),

        # Pair 6: Python Circular Imports
        (
            "Bagaimana cara mengatasi error 'ImportError: cannot import name X' akibat circular import di Python?",
            "Circular import terjadi ketika `module_a.py` meng-import komponen dari `module_b.py`, dan pada saat yang sama `module_b.py` meng-import komponen dari `module_a.py`. Hal ini membuat interpreter terjebak dalam inisialisasi modul yang belum selesai.\n\n"
            "**Cara Mengatasinya:**\n"
            "1. **Pindahkan import ke dalam fungsi (Lazy Import)**: Import hanya dilakukan saat fungsi dipanggil, bukan saat modul dimuat.\n"
            "2. **Refactoring (Ekstraksi dependensi)**: Buat modul baru `module_c.py` yang berisi komponen yang dibutuhkan oleh kedua modul tersebut.\n"
            "3. **Tulis ulang jenis import**: Ubah import absolut (`from a import x`) menjadi import modul langsung (`import a`).\n\n"
            "**Contoh Solusi 1 (Lazy Import):**\n"
            "```python\n"
            "# module_a.py\n"
            "# ❌ SALAH di tingkat modul:\n"
            "# from module_b import func_b\n"
            "\n"
            "def func_a():\n"
            "    # ✅ BENAR: Pindahkan import ke dalam fungsi\n"
            "    from module_b import func_b\n"
            "    print(\"Memanggil func_b dari func_a\")\n"
            "    func_b()\n"
            "```"
        ),

        # Pair 7: C++ Circular Dependency
        (
            "Bagaimana mengatasi circular dependency (ketergantungan melingkar) antar class di file header C++?",
            "Circular dependency terjadi ketika `class A` membutuhkan definisi `class B`, dan `class B` juga membutuhkan definisi `class A`.\n\n"
            "**Solusinya adalah menggunakan Forward Declaration** di header, dan memindahkan import header asli (`#include`) ke dalam file implementasi source (`.cpp`).\n\n"
            "**Contoh Kasus & Solusi:**\n"
            "```cpp\n"
            "// ── A.h ──────────────────────────────────────────────────────────\n"
            "#ifndef A_H\n"
            "#define A_H\n"
            "\n"
            "// ❌ JANGAN INCLUDE B.h di sini!\n"
            "// #include \"B.h\"\n"
            "\n"
            "class B; // ✅ Forward declaration: beritahu compiler bahwa class B akan ada\n"
            "\n"
            "class A {\n"
            "public:\n"
            "    B* b_member; // Gunakan pointer (*) atau reference (&), bukan objek langsung\n"
            "};\n"
            "\n"
            "#endif\n"
            "\n"
            "// ── A.cpp ────────────────────────────────────────────────────────\n"
            "#include \"A.h\"\n"
            "#include \"B.h\" // ✅ Taruh include asli di file .cpp\n"
            "\n"
            "void someFunc() {\n"
            "    // Di sini kita aman menggunakan member class B\n"
            "}\n"
            "```"
        ),

        # Pair 8: C++ Undefined Reference
        (
            "Mengapa saya mendapatkan linker error 'undefined reference to class::method()' di C++ padahal kodenya sudah dicompile?",
            "Linker error 'undefined reference' (atau `LNK2019` di MSVC) terjadi setelah kode sukses dicompile, tetapi linker tidak dapat menemukan implementasi biner dari fungsi atau class method yang dideklarasikan.\n\n"
            "**Penyebab & Solusi Umum:**\n"
            "1. **Lupa mengimplementasikan method**: Method telah dideklarasikan di `.h`, tetapi belum didefinisikan di `.cpp`.\n"
            "2. **File implementasi tidak ikut dicompile**: File `.cpp` tidak terdaftar di build system (misal lupa dimasukkan ke `CMakeLists.txt` atau target kompilasi g++).\n"
            "3. **Nama method/signature tidak cocok**: Perbedaan tipe data parameter atau penamaan namespace.\n\n"
            "**Verifikasi CMakeLists.txt:**\n"
            "```cmake\n"
            "# ❌ SALAH: File cpp baru lupa didaftarkan\n"
            "add_executable(my_app main.cpp)\n"
            "\n"
            "# ✅ BENAR: Pastikan file implementasi dicompile bersama\n"
            "add_executable(my_app main.cpp helper_class.cpp)\n"
            "```"
        ),

        # Pair 9: C++ Mutex & Race Condition
        (
            "Bagaimana cara mengatasi Race Condition saat membagikan data antar thread di C++ menggunakan Mutex?",
            "Race Condition terjadi ketika beberapa thread mengakses dan memodifikasi resource yang sama secara bersamaan tanpa sinkronisasi, mengakibatkan data korup atau inkonsisten.\n\n"
            "**Solusinya adalah mensinkronisasi akses dengan `std::mutex` dan `std::lock_guard` (RAII lock wrapper).**\n\n"
            "```cpp\n"
            "#include <thread>\n"
            "#include <mutex>\n"
            "#include <vector>\n"
            "\n"
            "int counter = 0;\n"
            "std::mutex counter_mutex; // Objek pengaman\n"
            "\n"
            "// ❌ SALAH: Mengakses variabel global secara bebas\n"
            "void unsafeIncrement() {\n"
            "    counter++; // Race condition terjadi di sini\n"
            "}\n"
            "\n"
            "// ✅ BENAR: Menggunakan lock_guard\n"
            "void safeIncrement() {\n"
            "    // Lock otomatis aktif di baris ini, dan dilepas saat keluar dari fungsi (destruktor)\n"
            "    std::lock_guard<std::mutex> lock(counter_mutex);\n"
            "    counter++;\n"
            "}\n"
            "\n"
            "void runThreads() {\n"
            "    std::vector<std::thread> threads;\n"
            "    for (int i = 0; i < 10; ++i) {\n"
            "        threads.emplace_back(safeIncrement);\n"
            "    }\n"
            "    for (auto& t : threads) t.join();\n"
            "}\n"
            "```"
        ),

        # Pair 10: JavaScript Async/Await loop performance
        (
            "Mengapa pemanggilan async/await di dalam loop 'for' terasa sangat lambat di JavaScript? Bagaimana mengoptimasinya?",
            "Menulis `await` di dalam loop `for` secara sequential (`for (const item of list) { await doSomething(item); }`) memaksa sistem untuk menunggu operasi saat ini selesai sebelum memulai operasi berikutnya.\n\n"
            "Jika operasi tersebut tidak bergantung satu sama lain, ini adalah pemborosan waktu. Kita harus menjalankannya secara parallel menggunakan `Promise.all()`.\n\n"
            "**Perbandingan Solusi:**\n"
            "```javascript\n"
            "const urls = ['/api/1', '/api/2', '/api/3'];\n"
            "\n"
            "// ❌ LAMBAT: Menjalankan request satu per satu (Sequential)\n"
            "async function fetchSequential() {\n"
            "    const results = [];\n"
            "    for (const url of urls) {\n"
            "        const res = await fetch(url); // Menunggu giliran\n"
            "        results.push(await res.json());\n"
            "    }\n"
            "    return results;\n"
            "}\n"
            "\n"
            "// ✅ CEPAT: Menjalankan semua request secara paralel\n"
            "async function fetchParallel() {\n"
            "    const promises = urls.map(async (url) => {\n"
            "        const res = await fetch(url);\n"
            "        return res.json();\n"
            "    });\n"
            "    // Tunggu semua request selesai bersamaan\n"
            "    return Promise.all(promises);\n"
            "}\n"
            "```"
        ),

        # Pair 11: Git Merge Conflict Resolution
        (
            "Bagaimana cara membaca format konflik merge Git (conflict markers) dan menyelesaikannya?",
            "Ketika Git mendeteksi modifikasi baris yang sama di dua branch berbeda saat proses merge, Git akan menandai area konflik tersebut di file sumber menggunakan penanda khusus:\n\n"
            "```text\n"
            "<<<<<<< HEAD\n"
            "kode di branch aktif Anda saat ini\n"
            "=======\n"
            "kode dari branch yang ingin Anda gabungkan\n"
            ">>>>>>> branch-name\n"
            "```\n\n"
            "**Langkah Penyelesaian:**\n"
            "1. Buka file yang berkonflik.\n"
            "2. Hapus penanda `<<<<<<<`, `=======`, dan `>>>>>>>`.\n"
            "3. Edit konten secara manual untuk menyatukan logika dari kedua versi.\n"
            "4. Lakukan staging dan commit file tersebut.\n\n"
            "**Contoh Resolusi:**\n"
            "```cpp\n"
            "// Sebelum:\n"
            "<<<<<<< HEAD\n"
            "int getPort() { return 8080; }\n"
            "=======\n"
            "int getPort() { return 9090; }\n"
            ">>>>>>> config-patch\n"
            "\n"
            "// Sesudah (resolusi manual memilih port 9090):\n"
            "int getPort() { return 9090; }\n"
            "```"
        ),

        # Pair 12: Rust error propagation
        (
            "Bagaimana cara melakukan error propagation yang clean di Rust menggunakan operator `?`?",
            "Daripada menggunakan blok `match` berulang-ulang untuk menangani enum `Result` di setiap langkah, Rust menyediakan operator `?` yang otomatis mengembalikan error jika bernilai `Err`, atau mengekstrak nilai sukses jika bernilai `Ok`.\n\n"
            "Fungsi yang menggunakan operator `?` harus mengembalikan tipe data yang kompatibel (seperti `Result` atau `Option`).\n\n"
            "**Contoh Kode:**\n"
            "```rust\n"
            "use std::fs::File;\n"
            "use std::io::{self, Read};\n"
            "\n"
            "// ❌ LEBIH PANJANG: Menggunakan match manual\n"
            "fn read_username_verbose() -> Result<String, io::Error> {\n"
            "    let f_res = File::open(\"username.txt\");\n"
            "    let mut f = match f_res {\n"
            "        Ok(file) => file,\n"
            "        Err(e) => return Err(e),\n"
            "    };\n"
            "    let mut s = String::new();\n"
            "    match f.read_to_string(&mut s) {\n"
            "        Ok(_) => Ok(s),\n"
            "        Err(e) => Err(e),\n"
            "    }\n"
            "}\n"
            "\n"
            "// ✅ BERSIH: Menggunakan operator `?` untuk propagasi cepat\n"
            "fn read_username_clean() -> Result<String, io::Error> {\n"
            "    let mut f = File::open(\"username.txt\")?; // Bubble up error if Err\n"
            "    let mut s = String::new();\n"
            "    f.read_to_string(&mut s)?;\n"
            "    Ok(s) // Sukses\n"
            "}\n"
            "```"
        ),

        # Pair 13: C++ Virtual Destructors
        (
            "Mengapa destructor base class harus ditandai sebagai `virtual` di C++?",
            "Jika Anda menghapus objek dari *derived class* (kelas turunan) melalui pointer ke *base class* (kelas induk) yang tidak memiliki destructor `virtual`, perilakunya adalah **undefined behavior**.\n\n"
            "Biasanya, ini mengakibatkan destructor kelas turunan tidak pernah dipanggil, menyebabkan memory leak pada resource yang dialokasikan di kelas turunan tersebut.\n\n"
            "**Contoh Masalah & Solusi:**\n"
            "```cpp\n"
            "// ❌ SALAH: Tanpa virtual destructor\n"
            "class BaseUnsafe {\n"
            "public:\n"
            "    ~BaseUnsafe() { std::cout << \"Base destructor\" << std::endl; }\n"
            "};\n"
            "\n"
            "class Derived : public BaseUnsafe {\n"
            "    int* m_data;\n"
            "public:\n"
            "    Derived() { m_data = new int[100]; }\n"
            "    ~Derived() { delete[] m_data; std::cout << \"Derived destructor\" << std::endl; }\n"
            "};\n"
            "\n"
            "void leak() {\n"
            "    BaseUnsafe* ptr = new Derived();\n"
            "    delete ptr; // ❌ HANYA memanggil ~BaseUnsafe(), m_data mengalami memory leak!\n"
            "}\n"
            "\n"
            "// ✅ BENAR: Gunakan virtual destructor\n"
            "class BaseSafe {\n"
            "public:\n"
            "    virtual ~BaseSafe() { std::cout << \"Base destructor\" << std::endl; } // virtual!\n"
            "};\n"
            "```"
        ),

        # Pair 14: C++ Volatile vs std::atomic
        (
            "Apa perbedaan antara `volatile` dan `std::atomic` di C++ untuk multithreading?",
            "Ini adalah kesalahpahaman umum. Keyword `volatile` di C++ **bukan** untuk sinkronisasi thread, melainkan untuk memberi tahu compiler agar tidak melakukan optimasi cache register pada variabel tersebut (karena nilainya dapat berubah di luar kontrol program, misalnya pada Memory-Mapped I/O hardware).\n\n"
            "`volatile` tidak memberikan garansi atomisitas dan tidak mencegah instruksi re-ordering oleh CPU.\n\n"
            "Untuk keamanan akses variabel di multithreading, Anda **harus** menggunakan `std::atomic`.\n\n"
            "**Perbandingan:**\n"
            "```cpp\n"
            "#include <atomic>\n"
            "\n"
            "// ❌ SALAH untuk multithreading: volatile tidak thread-safe\n"
            "volatile int unsafe_counter = 0;\n"
            "// Operasi counter++ pada volatile tidak bersifat atomis (terdiri dari read-modify-write)\n"
            "\n"
            "// ✅ BENAR: std::atomic menjamin operasi thread-safe bebas race condition\n"
            "std::atomic<int> safe_counter(0);\n"
            "// safe_counter++ aman diakses oleh banyak thread sekaligus secara konkuren\n"
            "```"
        ),

        # Pair 15: Rust Arc & Mutex
        (
            "Bagaimana cara membagikan status/data (shared mutable state) antar thread di Rust menggunakan `Arc` dan `Mutex`?",
            "Rust tidak mengizinkan peminjaman mutable biasa lintas thread secara bebas demi keamanan memori. Untuk membagikan data yang bisa dimodifikasi antar thread, Anda harus membungkus data tersebut menggunakan `Arc` (Atomic Reference Counted) dan `Mutex` (Mutual Exclusion).\n\n"
            "**Contoh Implementasi:**\n"
            "```rust\n"
            "use std::sync::{Arc, Mutex};\n"
            "use std::thread;\n"
            "\n"
            "fn main() {\n"
            "    // Bungkus counter dalam Mutex, lalu bungkus dalam Arc untuk multi-ownership\n"
            "    let counter = Arc::new(Mutex::new(0));\n"
            "    let mut handles = vec![];\n"
            "\n"
            "    for _ in 0..10 {\n"
            "        // Clone reference pointer-nya (bukan datanya)\n"
            "        let counter_clone = Arc::clone(&counter);\n"
            "        let handle = thread::spawn(move || {\n"
            "            // Kunci mutex sebelum mengakses data\n"
            "            let mut data = counter_clone.lock().unwrap();\n"
            "            *data += 1;\n"
            "            // Lock otomatis dilepas saat 'data' keluar scope\n"
            "        });\n"
            "        handles.push(handle);\n"
            "    }\n"
            "\n"
            "    for handle in handles {\n"
            "        handle.join().unwrap();\n"
            "    }\n"
            "\n"
            "    println!(\"Result: {}\", *counter.lock().unwrap()); // 10\n"
            "}\n"
            "```"
        ),

        # Pair 16: Python Generator Memory Efficiency
        (
            "Apa perbedaan performa memori antara list comprehension dan generator expression di Python?",
            "List comprehension `[x for x in range(n)]` membangun seluruh elemen list di memori secara langsung (eager evaluation).\n\n"
            "Generator expression `(x for x in range(n))` menggunakan *lazy evaluation*, memproses satu per satu elemen hanya ketika diminta (`__next__`). Ini menghemat memori secara masif saat menangani dataset besar.\n\n"
            "**Contoh Analisis Memori:**\n"
            "```python\n"
            "import sys\n"
            "\n"
            "n = 1000000\n"
            "\n"
            "# ❌ Memakan memori besar (~8 MB)\n"
            "list_comp = [x * 2 for x in range(n)]\n"
            "print(f\"List comp size: {sys.getsizeof(list_comp) / 1024 / 1024:.2f} MB\")\n"
            "\n"
            "# ✅ Sangat hemat memori (~100 bytes)\n"
            "gen_exp = (x * 2 for x in range(n))\n"
            "print(f\"Generator size: {sys.getsizeof(gen_exp)} bytes\")\n"
            "\n"
            "# Pemakaian fungsi yield untuk lazy-read file besar\n"
            "def read_huge_file(file_path):\n"
            "    with open(file_path) as f:\n"
            "        for line in f:\n"
            "            yield line.strip() # Tidak meload seluruh file ke RAM sekaligus\n"
            "```"
        ),

        # Pair 17: C/C++ Struct Padding & Alignment
        (
            "Apa itu struct padding di C/C++ dan bagaimana cara mengoptimasi susunan variabel dalam struct agar menghemat memori?",
            "Compiler C/C++ melakukan alignment memori pada struct berdasarkan ukuran tipe data terbesar anggotanya agar CPU dapat membacanya dengan performa optimal. Hal ini memicu disisipkannya byte kosong (padding) jika susunan variabel tidak efisien.\n\n"
            "**Contoh Susunan yang Buruk vs Optimal:**\n"
            "```cpp\n"
            "// ❌ KURANG EFISIEN: Ukuran = 12 bytes (pad 3 bytes + pad 1 byte)\n"
            "struct Unoptimized {\n"
            "    char a;     // 1 byte\n"
            "                // [3 bytes padding]\n"
            "    int b;      // 4 bytes\n"
            "    char c;     // 1 byte\n"
            "                // [3 bytes padding untuk kelipatan 4]\n"
            "};\n"
            "\n"
            "// ✅ OPTIMAL: Urutkan dari tipe data terbesar ke terkecil. Ukuran = 8 bytes\n"
            "struct Optimized {\n"
            "    int b;      // 4 bytes\n"
            "    char a;     // 1 byte\n"
            "    char c;     // 1 byte\n"
            "                // [2 bytes padding di akhir]\n"
            "};\n"
            "\n"
            "// Cetak perbandingan:\n"
            "// std::cout << sizeof(Unoptimized) << \" vs \" << sizeof(Optimized);\n"
            "```\n"
            "**Tips:** Gunakan attribute `#pragma pack(push, 1)` jika Anda wajib mematikan padding sepenuhnya (misal untuk serialisasi struktur paket jaringan)."
        ),

        # Pair 18: Go Channels & Synchronization
        (
            "Bagaimana cara sinkronisasi goroutine menggunakan Channels di bahasa pemrograman Go?",
            "Di Go, goroutine dijalankan secara asinkron. Untuk mengirim data atau mensinkronisasi selesainya tugas antar goroutine, gunakan channel.\n\n"
            "**Contoh Sinkronisasi Sederhana:**\n"
            "```go\n"
            "package main\n"
            "\n"
            "import (\n"
            "\t\"fmt\"\n"
            "\t\"time\"\n"
            ")\n"
            "\n"
            "func worker(done chan bool) {\n"
            "\tfmt.Println(\"Memulai pekerjaan...\")\n"
            "\ttime.Sleep(time.Second) // Simulasi kerja berat\n"
            "\tfmt.Println(\"Pekerjaan selesai!\")\n"
            "\n"
            "\t// Kirim sinyal selesai ke channel\n"
            "\tdone <- true\n"
            "}\n"
            "\n"
            "func main() {\n"
            "\t// Buat buffered channel bertipe boolean\n"
            "\tdone := make(chan bool, 1)\n"
            "\n"
            "\tgo worker(done) // Jalankan goroutine\n"
            "\n"
            "\t// Blok eksekusi main sampai sinyal masuk dari channel done\n"
            "\t<-done \n"
            "\tfmt.Println(\"Program utama ditutup.\")\n"
            "}\n"
            "```"
        ),

        # Pair 19: Python Context Managers
        (
            "Bagaimana cara membuat custom context manager di Python menggunakan class atau generator decorator?",
            "Context manager (`with` statement) mengamankan akuisisi dan pembebasan resource secara otomatis (mirip konsep RAII).\n\n"
            "**Metode 1: Berbasis Class (`__enter__` & `__exit__`)**\n"
            "```python\n"
            "class DatabaseConnection:\n"
            "    def __enter__(self):\n"
            "        print(\"Membuka koneksi database...\")\n"
            "        return self\n"
            "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
            "        print(\"Menutup koneksi database...\")\n"
            "        # return True untuk mensupress exception jika diinginkan\n"
            "\n"
            "with DatabaseConnection() as db:\n"
            "    print(\"Menulis data...\")\n"
            "```\n\n"
            "**Metode 2: Berbasis Generator (`@contextmanager`)**\n"
            "```python\n"
            "from contextlib import contextmanager\n"
            "\n"
            "@contextmanager\n"
            "def open_file_safe(path):\n"
            "    print(\"Membuka file...\")\n"
            "    f = open(path, 'w')\n"
            "    try:\n"
            "        yield f\n"
            "    finally:\n"
            "        print(\"Menutup file secara aman di blok finally...\")\n"
            "        f.close()\n"
            "\n"
            "with open_file_safe(\"log.txt\") as f:\n"
            "    f.write(\"Moko system logged.\\n\")\n"
            "```"
        ),

        # Pair 20: Modern CMake Target Best Practices
        (
            "Mengapa disarankan menggunakan target-based CMake daripada global variables?",
            "CMake modern menggunakan prinsip Object-Oriented/Target-based. Daripada menetapkan direktori include dan pustaka link secara global (yang memicu polusi dependensi ke target lain), kita mengikat dependensi langsung ke target yang bersangkutan.\n\n"
            "**Perbandingan Praktik:**\n"
            "```cmake\n"
            "# ❌ PRAKTIK BURUK (Global Variables - Legacy CMake):\n"
            "include_directories(${QT_INCLUDE_DIRS})\n"
            "link_libraries(${QT_LIBRARIES})\n"
            "add_executable(app main.cpp)\n"
            "\n"
            "# ✅ PRAKTIK BAIK (Target-based - Modern CMake):\n"
            "add_executable(moko_app main.cpp)\n"
            "\n"
            "# Hubungkan include path hanya ke target moko_app\n"
            "target_include_directories(moko_app PRIVATE ${QT_INCLUDE_DIRS})\n"
            "\n"
            "# Hubungkan target link library secara eksklusif\n"
            "target_link_libraries(moko_app PRIVATE Qt5::Widgets)\n"
            "```"
        ),

        # Pair 21: SQL Injection Prevention
        (
            "Bagaimana cara mencegah celah keamanan SQL Injection di kode backend aplikasi?",
            "SQL Injection terjadi saat input mentah dari user digabungkan langsung dengan string query database, sehingga SQL interpreter mengeksekusi perintah jahat.\n\n"
            "**Cara Pencegahan:** Gunakan **Prepared Statements / Parameterized Queries**. Ini memisahkan query template dari parameter input user.\n\n"
            "```python\n"
            "import sqlite3\n"
            "\n"
            "conn = sqlite3.connect(\"moko.db\")\n"
            "cursor = conn.cursor()\n"
            "\n"
            "user_input = \"admin' OR '1'='1\"\n"
            "\n"
            "# ❌ RENTAN CELEH: Menggunakan string formatting\n"
            "unsafe_query = f\"SELECT * FROM users WHERE username = '{user_input}'\"\n"
            "cursor.execute(unsafe_query)  # Celah terbuka!\n"
            "\n"
            "# ✅ AMAN: Menggunakan Parameterized Query\n"
            "safe_query = \"SELECT * FROM users WHERE username = ?\"\n"
            "cursor.execute(safe_query, (user_input,))  # Input otomatis di-escape database engine\n"
            "```"
        ),
    ]

    for user, assistant in pairs:
        samples.append(make_sample(user, assistant,
            metadata={"source": "programming_errors", "type": "debugging"}))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    log("=" * 60)
    log("MOKO Data Factory v1.0 — Starting Generation")
    log("=" * 60)

    generators = [
        ("C++/Qt5 Patterns",   generate_cpp_qt_samples,       OUT_CPP_QT),
        ("Algorithms & DS",    generate_algorithm_samples,    OUT_ALGO),
        ("Security Hacking",   generate_security_samples,     OUT_SECURITY),
        ("MOKO Integration",   generate_integration_samples,  OUT_INTEGRATION),
        ("Multi-turn Debug",   generate_multiturn_samples,    OUT_MULTITURN),
        ("Chain-of-Thought",   generate_reasoning_samples,    OUT_REASONING),
        ("Docs -> Q&A",        generate_docs_samples,         OUT_DOCS),
        ("Programming & Debug",generate_programming_error_samples, OUT_PROGRAMMING),
    ]

    total = 0
    for name, fn, out in generators:
        log(f"Generating: {name}...")
        samples = fn()
        written = dedup_and_write(samples, out, append=True)
        total  += written
        log(f"  -> {written:,} new samples -> {out.name}", "OK")

    log("=" * 60)
    log(f"TOTAL NEW SAMPLES WRITTEN: {total:,}", "OK")

    # Summary
    print()
    log("Dataset Summary:")
    grand = 0
    for f in sorted(DATASET_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(f, encoding='utf-8') if _.strip())
        grand += count
        kb    = f.stat().st_size // 1024
        print(f"  {f.name:<42} {count:>6,} samples  ({kb:,} KB)")
    print(f"  {'GRAND TOTAL':<42} {grand:>6,} samples")
    log("=" * 60)
    return grand


def validate_all():
    log("Validating all JSONL files...")
    for f in sorted(DATASET_DIR.glob("*.jsonl")):
        valid = broken = 0
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                    assert 'messages' in d and len(d['messages']) >= 2
                    valid += 1
                except:
                    broken += 1
        lvl = "OK" if broken == 0 else "WARN"
        log(f"  {f.name}: valid={valid} broken={broken}", lvl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOKO Data Factory v1.0")
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--stats",    action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate_all()
    elif args.stats:
        print("Dataset Stats:")
        total = 0
        for f in sorted(DATASET_DIR.glob("*.jsonl")):
            count = sum(1 for _ in open(f, encoding='utf-8') if _.strip())
            total += count
            print(f"  {f.name:<40} {count:>6,}")
        print(f"  {'TOTAL':<40} {total:>6,}")
    else:
        run_all()
