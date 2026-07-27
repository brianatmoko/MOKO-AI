#include "graphify_dialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QTextBrowser>
#include <QGraphicsSceneHoverEvent>
#include <QPainter>
#include <QPainterPath>
#include <QGraphicsSceneMouseEvent>
#include <QDirIterator>
#include <QFile>
#include <QTextStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <cmath>
#include <QDebug>
#include <QDir>
#include <QSet>
#include <numeric>
#include <algorithm>
#include <random>

// ──────────────────────────────────────────────────────────────────────────────
// Stylesheet
// ──────────────────────────────────────────────────────────────────────────────
static const QString GRAPHIFY_QSS = R"(
QDialog {
    background-color: #03030a;
    color: #d0d0e0;
    font-family: 'Inter', 'Fira Code', 'Segoe UI', sans-serif;
}
QFrame#panel_card {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #08080f, stop:1 #050509);
    border: 1px solid #111120;
    border-radius: 12px;
}
QLabel { color: #7a7a98; font-size: 11px; }
QLabel#header_title { color: #00ffaa; font-weight: bold; font-size: 13px; letter-spacing: 2px; }
QLabel#stat_val     { color: #00ffaa; font-weight: bold; font-size: 13px; }
QPushButton {
    background: rgba(0,255,170,0.05);
    border: 1px solid rgba(0,255,170,0.18);
    border-radius: 6px;
    color: #00ffaa;
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    padding: 6px 14px;
    min-height: 28px;
}
QPushButton:hover { background: rgba(0,255,170,0.14); border-color:#00ffaa; color:#fff; }
QLineEdit {
    background: #020208;
    border: 1px solid #111120;
    border-radius: 6px;
    color: #fff;
    padding: 6px 12px;
    font-size: 11px;
    font-family: 'Fira Code', monospace;
}
QLineEdit:focus { border-color: #00ffaa; }
QTextBrowser {
    background: #020208;
    border: 1px solid #0e0e1a;
    border-radius: 8px;
    color: #a0a0c0;
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    line-height: 1.4;
}
)";

// ──────────────────────────────────────────────────────────────────────────────
// NeuronNodeItem — Tiny glowing dot, NO circles
// ──────────────────────────────────────────────────────────────────────────────

NeuronNodeItem::NeuronNodeItem(const QString& id, const QString& label, NodeType type, const QColor& color)
    : m_id(id), m_label(label), m_type(type), m_color(color), m_pos(0, 0)
{
    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
}

QRectF NeuronNodeItem::boundingRect() const {
    double r = (m_type == DomainCell) ? 3.5 : (m_type == BucketCell) ? 1.8 : 1.0;
    double extra = m_hovered ? 85.0 : 8.0;
    return QRectF(-r - extra, -r - extra, 2*(r + extra), 2*(r + extra));
}

void NeuronNodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    painter->setRenderHint(QPainter::Antialiasing);

    // ── Dot radius by type (ultra-tiny points, no visible circles) ──────────
    double r = (m_type == DomainCell) ? 3.5 : (m_type == BucketCell) ? 1.8 : 1.0;

    // Subtle soft glow behind dot (only domain or hovered)
    if (m_type == DomainCell || m_hovered) {
        QColor glow = m_color;
        glow.setAlpha(m_hovered ? 70 : 25);
        painter->setBrush(glow);
        painter->setPen(Qt::NoPen);
        painter->drawEllipse(QPointF(0,0), r + 2.0, r + 2.0);
    }

    // Core dot — solid bright point
    QColor core = m_hovered ? Qt::white : m_color;
    if (!m_hovered && m_type == MemoryCell) {
        core = m_color;
        core.setAlpha(180);
    }
    painter->setBrush(core);
    painter->setPen(Qt::NoPen);
    painter->drawEllipse(QPointF(0,0), r, r);

    // Label (domain only, or on hover)
    if (m_type == DomainCell || m_hovered) {
        QFont f = painter->font();
        f.setPointSize(m_type == DomainCell ? 8 : 7);
        f.setBold(m_type == DomainCell);
        painter->setFont(f);

        QColor txtCol = m_hovered ? Qt::white : QColor(200, 200, 220);
        painter->setPen(txtCol);
        // Draw label below the dot
        painter->drawText(QRectF(-75, r + 3, 150, 16), Qt::AlignCenter, m_label);
    }
}

void NeuronNodeItem::hoverEnterEvent(QGraphicsSceneHoverEvent* e) {
    Q_UNUSED(e);
    m_hovered = true;
    prepareGeometryChange();
    update();
}

void NeuronNodeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* e) {
    Q_UNUSED(e);
    m_hovered = false;
    prepareGeometryChange();
    update();
}

void NeuronNodeItem::mousePressEvent(QGraphicsSceneMouseEvent* e) {
    QGraphicsItem::mousePressEvent(e);
}

// ──────────────────────────────────────────────────────────────────────────────
// NeuronEdgeItem — Ultra-thin thread fiber
// ──────────────────────────────────────────────────────────────────────────────

NeuronEdgeItem::NeuronEdgeItem(NeuronNodeItem* src, NeuronNodeItem* dst, const QColor& color, double weight)
    : m_src(src), m_dst(dst), m_color(color), m_weight(weight) {}

QRectF NeuronEdgeItem::boundingRect() const {
    if (!m_src || !m_dst) return QRectF();
    return QRectF(m_src->pos(), m_dst->pos()).normalized().adjusted(-20, -20, 20, 20);
}

void NeuronEdgeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    if (!m_src || !m_dst) return;
    painter->setRenderHint(QPainter::Antialiasing);

    QPointF p1 = m_src->pos();
    QPointF p2 = m_dst->pos();
    QPointF diff = p2 - p1;
    double len = std::sqrt(diff.x()*diff.x() + diff.y()*diff.y());
    if (len < 1.0) return;

    // Slight organic curve (nerve fiber, not straight line)
    QPointF mid = (p1 + p2) * 0.5;
    QPointF normal(-diff.y() / len, diff.x() / len);
    double curveOffset = len * 0.10;  // gentle, not dramatic
    QPointF ctrl = mid + normal * curveOffset;

    QPainterPath path;
    path.moveTo(p1);
    path.quadTo(ctrl, p2);

    // Thread color
    QColor penColor = m_color;
    if (m_highlighted) {
        penColor = QColor(0, 255, 170);
        penColor.setAlpha(220);
    } else {
        // Cross-domain threads slightly brighter
        int alpha = (m_weight >= 0.8) ? 50 : 30;
        penColor.setAlpha(alpha);
    }

    // Ultra-thin thread
    double thickness = m_highlighted ? 0.9 : 0.4;
    QPen pen(penColor, thickness);
    pen.setCapStyle(Qt::RoundCap);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(path);

    // Travelling signal pulse along thread
    double phase = std::fmod(m_pulsePhase + m_phaseOffset, 1.0);
    QPointF pulsePos = path.pointAtPercent(phase);
    QColor pulseCol = m_highlighted ? Qt::white : m_color.lighter(160);
    pulseCol.setAlpha(m_highlighted ? 255 : 140);
    painter->setBrush(pulseCol);
    painter->setPen(Qt::NoPen);
    double pr = m_highlighted ? 1.8 : 1.0;
    painter->drawEllipse(pulsePos, pr, pr);
}

// ──────────────────────────────────────────────────────────────────────────────
// NeuronScene
// ──────────────────────────────────────────────────────────────────────────────

NeuronScene::NeuronScene(QObject* parent) : QGraphicsScene(parent) {
    setSceneRect(-900, -900, 1800, 1800);
}

void NeuronScene::addNeuron(NeuronNodeItem* node) {
    m_nodes.append(node);
    addItem(node);
}

void NeuronScene::addFiber(NeuronEdgeItem* edge) {
    m_edges.append(edge);
    addItem(edge);
}

void NeuronScene::clearAll() {
    clear();
    m_nodes.clear();
    m_edges.clear();
}

void NeuronScene::highlightPathway(const QString& nodeId) {
    for (auto* e : m_edges) {
        e->m_highlighted = (e->m_src->m_id == nodeId || e->m_dst->m_id == nodeId
                            || e->m_src->m_id.startsWith(nodeId)
                            || e->m_dst->m_id.startsWith(nodeId));
    }
    update();
}

void NeuronScene::animateNetwork() {
    m_pulsePhase = std::fmod(m_pulsePhase + 0.006, 1.0);
    for (auto* e : m_edges) {
        e->setPulsePhase(m_pulsePhase);
    }
    update();
}

// ──────────────────────────────────────────────────────────────────────────────
// MokoGraphifyDialog
// ──────────────────────────────────────────────────────────────────────────────

MokoGraphifyDialog::MokoGraphifyDialog(QWidget* parent)
    : QDialog(parent)
{
    m_omniRoot = QDir::homePath() + "/Documents/Linux/MOKO_OS_Project/.moko_omni";

    setWindowTitle("🌐 MOKO Graphify — Neural Memory Threads");
    resize(1180, 860);
    setStyleSheet(GRAPHIFY_QSS);

    buildUi();

    m_animTimer = new QTimer(this);
    connect(m_animTimer, &QTimer::timeout, m_scene, &NeuronScene::animateNetwork);
    m_animTimer->start(30);

    refreshData();
}

MokoGraphifyDialog::~MokoGraphifyDialog() {
    m_animTimer->stop();
}

void MokoGraphifyDialog::buildUi() {
    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(14, 14, 14, 14);
    mainLayout->setSpacing(12);

    // ── LEFT SIDEBAR ─────────────────────────────────────────────────────────
    QVBoxLayout* sideLayout = new QVBoxLayout();
    sideLayout->setSpacing(12);

    QFrame* statCard = new QFrame(this);
    statCard->setObjectName("panel_card");
    statCard->setFixedWidth(270);
    QVBoxLayout* statLay = new QVBoxLayout(statCard);
    statLay->setContentsMargins(14,14,14,14);
    statLay->setSpacing(10);

    QLabel* title = new QLabel("⬡ GRAPHIFY THREADS", statCard);
    title->setObjectName("header_title");
    statLay->addWidget(title);

    auto makeStatRow = [&](const QString& lbl, QLabel*& valLabel) {
        QHBoxLayout* row = new QHBoxLayout();
        QLabel* l = new QLabel(lbl, statCard);
        valLabel = new QLabel("—", statCard);
        valLabel->setObjectName("stat_val");
        row->addWidget(l);
        row->addWidget(valLabel, 0, Qt::AlignRight);
        return row;
    };

    statLay->addLayout(makeStatRow("Total Memori (RAG):",  m_lblTotalMemories));
    statLay->addLayout(makeStatRow("Domain Aktif:",        m_lblActiveDomains));
    statLay->addLayout(makeStatRow("Bucket Aktif:",        m_lblActiveBuckets));
    statLay->addSpacing(8);

    QLabel* lblDist = new QLabel("Distribusi Domain:", statCard);
    lblDist->setStyleSheet("font-weight:bold; color:#00ffaa;");
    statLay->addWidget(lblDist);

    m_txtStats = new QTextBrowser(statCard);
    m_txtStats->setMinimumHeight(100);
    m_txtStats->setMaximumHeight(140);
    statLay->addWidget(m_txtStats);

    QLabel* lblQ = new QLabel("Cari Jalur Semantik:", statCard);
    lblQ->setStyleSheet("font-weight:bold; color:#00ffaa;");
    statLay->addWidget(lblQ);

    QHBoxLayout* searchRow = new QHBoxLayout();
    m_txtSearch = new QLineEdit(statCard);
    m_txtSearch->setPlaceholderText("query...");
    m_btnSearch = new QPushButton("Sorot", statCard);
    connect(m_btnSearch, &QPushButton::clicked, this, &MokoGraphifyDialog::runSearch);
    searchRow->addWidget(m_txtSearch, 1);
    searchRow->addWidget(m_btnSearch);
    statLay->addLayout(searchRow);

    QPushButton* btnRefresh = new QPushButton("🔄 Refresh", statCard);
    connect(btnRefresh, &QPushButton::clicked, this, &MokoGraphifyDialog::refreshData);
    statLay->addWidget(btnRefresh);

    sideLayout->addWidget(statCard);

    // Inspector
    QFrame* detailCard = new QFrame(this);
    detailCard->setObjectName("panel_card");
    detailCard->setFixedWidth(270);
    QVBoxLayout* detLay = new QVBoxLayout(detailCard);
    detLay->setContentsMargins(14,14,14,14);

    QLabel* detTitle = new QLabel("🔍 Inspektur Node:", detailCard);
    detTitle->setStyleSheet("font-weight:bold; color:#00ffaa;");
    detLay->addWidget(detTitle);

    m_txtDetails = new QTextBrowser(detailCard);
    m_txtDetails->setMinimumHeight(120);
    m_txtDetails->setMaximumHeight(200);
    m_txtDetails->setPlaceholderText("Klik atau hover titik untuk melihat info...");
    detLay->addWidget(m_txtDetails);

    sideLayout->addWidget(detailCard);

    // Legend
    QFrame* legendCard = new QFrame(this);
    legendCard->setObjectName("panel_card");
    legendCard->setFixedWidth(270);
    QVBoxLayout* legLay = new QVBoxLayout(legendCard);
    legLay->setContentsMargins(14,14,14,14);
    legLay->setSpacing(6);
    QLabel* legTitle = new QLabel("📋 LEGENDA BENANG", legendCard);
    legTitle->setStyleSheet("font-weight:bold; color:#00ffaa; font-size:11px;");
    legLay->addWidget(legTitle);
    QTextBrowser* legView = new QTextBrowser(legendCard);
    legView->setMaximumHeight(130);
    legView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    legView->setHtml(R"(
<style>
 body{background:#020208;color:#b0b0c0;font-family:'Fira Code',monospace;font-size:10px;margin:2px;padding:0;}
 .row{padding:4px 0;border-bottom:1px solid #0e0e18;}
</style>
<div class='row'><span style='color:#00ff78;font-size:13px'>•</span>&nbsp; Titik Domain (pusat kognitif)</div>
<div class='row'><span style='color:#d246ff;font-size:10px'>·</span>&nbsp; Titik Bucket (index SimHash)</div>
<div class='row'><span style='color:#00e6ff;font-size:8px'>·</span>&nbsp; Titik Memori (rekaman SHA3)</div>
<div class='row'><span style='color:#333;'>───</span>&nbsp; Benang aksion (tipis)</div>
<div class='row'><span style='color:#555;'>- -</span>&nbsp; Benang lintas domain</div>
<div class='row'><span style='color:#00ffaa;font-size:10px'>·</span>&nbsp; Pulsa sinyal (animasi)</div>
)");
    legLay->addWidget(legView);
    sideLayout->addWidget(legendCard);
    sideLayout->addStretch();

    mainLayout->addLayout(sideLayout);

    // ── GRAPH VIEW ──────────────────────────────────────────────────────────
    m_scene = new NeuronScene(this);
    m_view  = new QGraphicsView(m_scene, this);
    m_view->setRenderHint(QPainter::Antialiasing);
    m_view->setRenderHint(QPainter::SmoothPixmapTransform);
    m_view->setDragMode(QGraphicsView::ScrollHandDrag);
    m_view->setBackgroundBrush(QColor(3, 3, 10));
    m_view->setStyleSheet("border: 1px solid #111120; border-radius: 12px;");
    m_view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    connect(m_scene, &QGraphicsScene::selectionChanged, this, [this]() {
        auto sel = m_scene->selectedItems();
        if (!sel.isEmpty()) onNodeSelected(sel.first());
    });

    mainLayout->addWidget(m_view, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// scanOmniStore — Reads first line of each meta.jsonl up to MAX cap
// ──────────────────────────────────────────────────────────────────────────────
void MokoGraphifyDialog::scanOmniStore() {
    // Deprecated: use buildDomainBucketMap() instead.
    // Kept for MemoryRecord compatibility (optional meta loading).
    m_records.clear();
    // (no-op — bucket discovery now done inside refreshData directly)
}

// ──────────────────────────────────────────────────────────────────────────────
// refreshData — Thread-based graph layout (dots + fibers, no circles)
// ──────────────────────────────────────────────────────────────────────────────
void MokoGraphifyDialog::refreshData() {
    m_scene->clearAll();

    // ── 1. Scan all domains and their active buckets directly from disk ────────
    // We enumerate l1/l2 directory pairs and confirm via index.bin.
    // We do NOT require meta.jsonl — so code/security/math all show up.
    // -------------------------------------------------------------------------
    // Structure: .moko_omni/<domain>/<l1>/<l2>/index.bin
    // -------------------------------------------------------------------------

    struct BucketInfo {
        QString folder;    // "a880/c544"
        int     count;     // entries in index.bin
    };

    QMap<QString, int>              domainTrueCounts;
    QMap<QString, int>              domainTrueBuckets;
    QMap<QString, QList<BucketInfo>> domainBucketList;  // domain → sampled buckets

    const int MAX_BUCKETS_PER_DOMAIN = 35;  // visual nodes per domain
    int grandTotalMemories = 0;
    int grandTotalBuckets  = 0;

    QDir rootDir(m_omniRoot);
    if (!rootDir.exists()) return;

    for (const QString& dName : rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        QDir domDir(rootDir.filePath(dName));
        int domMem = 0, domBuck = 0;
        QList<BucketInfo> sampledBuckets;

        // Evenly sample buckets across all l1 subdirs for visual spread
        QStringList l1Names = domDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        int nL1 = l1Names.size();
        // Stride: pick evenly-spaced l1 dirs to get spatial diversity
        int stride = qMax(1, nL1 / MAX_BUCKETS_PER_DOMAIN);
        int visLoaded = 0;

        for (int li = 0; li < nL1; ++li) {
            QDir bDir(domDir.filePath(l1Names[li]));
            QStringList l2Names = bDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

            for (const QString& sbName : l2Names) {
                QFile idxFile(bDir.filePath(sbName + "/index.bin"));
                if (!idxFile.exists()) continue;
                int n = idxFile.size() / 44;
                if (n <= 0) continue;

                domMem  += n;
                domBuck += 1;

                // Visual sampling: take one bucket per stride stride across l1
                if (visLoaded < MAX_BUCKETS_PER_DOMAIN && (li % stride == 0)) {
                    sampledBuckets.append({ l1Names[li] + "/" + sbName, n });
                    visLoaded++;
                }
            }
        }

        if (domMem > 0) {
            domainTrueCounts[dName]  = domMem;
            domainTrueBuckets[dName] = domBuck;
            domainBucketList[dName]  = sampledBuckets;
            grandTotalMemories += domMem;
            grandTotalBuckets  += domBuck;
        }
    }

    m_lblTotalMemories->setText(QString::number(grandTotalMemories));
    m_lblActiveDomains->setText(QString::number(domainTrueCounts.size()));
    m_lblActiveBuckets->setText(QString::number(grandTotalBuckets));

    QString distHtml;
    for (auto it = domainTrueCounts.begin(); it != domainTrueCounts.end(); ++it) {
        distHtml += QString("<span style='color:#00ffaa;'>⬡</span> %1: <b>%2</b> / %3 bkt<br>")
            .arg(it.key()).arg(it.value()).arg(domainTrueBuckets.value(it.key(), 0));
    }
    m_txtStats->setHtml(distHtml);

    // ── 3. Colour palette per domain ─────────────────────────────────────────
    QMap<QString, QColor> domColors;
    domColors["general"]     = QColor(  0, 220, 255); // Cyan
    domColors["code"]        = QColor(180,  60, 255); // Electric violet
    domColors["math"]        = QColor(  0, 255, 120); // Radiant green
    domColors["science"]     = QColor(255, 165,   0); // Amber
    domColors["history"]     = QColor(150,   0, 255); // Purple
    domColors["finance"]     = QColor(255, 210,   0); // Gold
    domColors["health"]      = QColor(255,  80,  80); // Rose
    domColors["security"]    = QColor(255,  30,  60); // Crimson
    domColors["programming"] = QColor(  0, 200, 180); // Teal

    // ── 4. Domain nodes on outer ring ────────────────────────────────────────
    QStringList domList = domainTrueCounts.keys();
    int nDom = domList.size();
    if (nDom == 0) return;

    // Canvas layout: domains on a circle of radius 620px
    const double DOMAIN_RADIUS = 620.0;

    // Sectors — each domain gets a 360/nDom degree wedge of the canvas.
    // Buckets are placed inside their domain's wedge.
    QMap<QString, NeuronNodeItem*> domainNodeMap;

    // Seeded RNG for reproducible jitter (same layout every refresh)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-18.0, 18.0);

    for (int i = 0; i < nDom; ++i) {
        const QString& dom   = domList[i];
        QColor col           = domColors.value(dom, QColor(80, 200, 200));
        // Domain centre angle (evenly spaced, start from top = -π/2)
        double domAngle      = (2.0 * M_PI * i) / nDom - M_PI / 2.0;
        double sectorHalf    = M_PI / nDom;  // half-width of each sector

        QPointF domPos(std::cos(domAngle) * DOMAIN_RADIUS,
                       std::sin(domAngle) * DOMAIN_RADIUS);

        NeuronNodeItem* domNode = new NeuronNodeItem(dom, dom.toUpper(),
                                                      NeuronNodeItem::DomainCell, col);
        domNode->m_pos = domPos;
        domNode->setPos(domPos);
        m_scene->addNeuron(domNode);
        domainNodeMap[dom] = domNode;

        // ── Buckets spread across this domain's angular sector ────────────
        const QList<BucketInfo>& buckets = domainBucketList.value(dom);
        int nBuck = buckets.size();
        if (nBuck == 0) continue;

        // Golden-angle spread within sector
        const double PHI     = 2.399963; // golden angle in radians
        const double B_INNER = 160.0;
        const double B_OUTER = 530.0;

        for (int j = 0; j < nBuck; ++j) {
            const BucketInfo& bk = buckets[j];

            // Normalised position [0, 1]
            double t = (nBuck > 1) ? (double)j / (nBuck - 1) : 0.5;

            // Radius: spread from inner to outer
            double bRadius = B_INNER + t * (B_OUTER - B_INNER);

            // Angle: fill entire sector width, plus golden-angle micro-jitter
            double bAngle = domAngle + sectorHalf * (2.0 * t - 1.0);
            bAngle += std::fmod(PHI * j, sectorHalf * 0.35);

            QPointF buckPos(std::cos(bAngle) * bRadius + jitter(rng),
                            std::sin(bAngle) * bRadius + jitter(rng));

            QString bId = dom + ":" + bk.folder;
            NeuronNodeItem* buckNode = new NeuronNodeItem(bId, bk.folder.right(4),
                                                          NeuronNodeItem::BucketCell,
                                                          col.darker(115));
            buckNode->m_pos = buckPos;
            buckNode->setPos(buckPos);
            m_scene->addNeuron(buckNode);

            // Thread: domain node → bucket node
            NeuronEdgeItem* fiber = new NeuronEdgeItem(domNode, buckNode, col);
            fiber->m_phaseOffset  = j * 0.14;
            m_scene->addFiber(fiber);

            // ── Tiny memory endpoint dots around the bucket ───────────────
            // Represent the count of memories (1 dot per 20% of max, capped at 5)
            int maxCount = domainTrueCounts.value(dom, 1);
            int nDots = qMin(5, qMax(1, (bk.count * 5) / qMax(1, maxCount / nBuck)));
            for (int k = 0; k < nDots; ++k) {
                double mAngle  = bAngle + (k - (nDots-1)*0.5) * 0.14;
                double mRadius = bRadius + 30.0 + (k % 2) * 16.0;

                QPointF memPos(std::cos(mAngle) * mRadius + jitter(rng) * 0.4,
                               std::sin(mAngle) * mRadius + jitter(rng) * 0.4);

                QString memId = bId + "_" + QString::number(k);
                NeuronNodeItem* memNode = new NeuronNodeItem(
                    memId, "", NeuronNodeItem::MemoryCell, col.lighter(150));
                memNode->m_pos = memPos;
                memNode->setPos(memPos);
                m_scene->addNeuron(memNode);

                // Thread: bucket → memory endpoint
                NeuronEdgeItem* synapse = new NeuronEdgeItem(buckNode, memNode,
                                                              col.lighter(120), 0.45);
                synapse->m_phaseOffset = (j + k) * 0.08;
                m_scene->addFiber(synapse);
            }
        }
    }

    // ── 5. Cross-domain threads (connect related domain nodes) ────────────────
    // These are the long inter-domain threads spanning the full canvas
    struct CrossLink { QString a, b; };
    QList<CrossLink> links = {
        {"code",     "math"},
        {"code",     "security"},
        {"code",     "general"},
        {"math",     "science"},
        {"finance",  "math"},
        {"security", "general"},
        {"history",  "general"},
        {"programming","code"},
    };

    for (const auto& lnk : links) {
        if (!domainNodeMap.contains(lnk.a) || !domainNodeMap.contains(lnk.b)) continue;

        NeuronNodeItem* nA = domainNodeMap[lnk.a];
        NeuronNodeItem* nB = domainNodeMap[lnk.b];

        QColor midCol(
            (nA->m_color.red()   + nB->m_color.red())   / 2,
            (nA->m_color.green() + nB->m_color.green()) / 2,
            (nA->m_color.blue()  + nB->m_color.blue())  / 2
        );
        midCol.setAlpha(60);

        NeuronEdgeItem* crossFiber = new NeuronEdgeItem(nA, nB, midCol, 0.9);
        crossFiber->m_phaseOffset  = 0.5;
        m_scene->addFiber(crossFiber);
    }

    // Fit view to content
    m_view->fitInView(m_scene->itemsBoundingRect().adjusted(-60,-60,60,60),
                      Qt::KeepAspectRatio);
}

// ──────────────────────────────────────────────────────────────────────────────
void MokoGraphifyDialog::onNodeSelected(QGraphicsItem* item) {
    NeuronNodeItem* node = dynamic_cast<NeuronNodeItem*>(item);
    if (!node) return;

    m_scene->highlightPathway(node->m_id);

    QString html;
    html += QString("<b style='color:%1;font-size:12px;'>◉ %2</b><br>")
        .arg(node->m_color.name(), node->m_label);

    if (node->m_type == NeuronNodeItem::DomainCell) {
        html += "<br><b>Tipe:</b> Domain Kognitif<br>";
        html += QString("<b>ID:</b> %1").arg(node->m_id);
    } else if (node->m_type == NeuronNodeItem::BucketCell) {
        html += "<br><b>Tipe:</b> Bucket SimHash<br>";
        html += QString("<b>Path:</b> %1").arg(node->m_id);
    } else {
        html += "<br><b>Tipe:</b> Rekaman Memori SHA3<br>";
        html += QString("<b>Hash:</b> %1").arg(node->m_id.left(16) + "…");
    }
    m_txtDetails->setHtml(html);
}

// ──────────────────────────────────────────────────────────────────────────────
void MokoGraphifyDialog::runSearch() {
    QString q = m_txtSearch->text().trimmed().toLower();
    if (q.isEmpty()) return;

    QString target = "general";
    if (q.contains("code") || q.contains("python") || q.contains("cpp") || q.contains("java"))
        target = "code";
    else if (q.contains("math") || q.contains("matrix") || q.contains("hitung"))
        target = "math";
    else if (q.contains("security") || q.contains("hack") || q.contains("exploit")
             || q.contains("injection") || q.contains("xss") || q.contains("sql")
             || q.contains("pentest") || q.contains("keamanan") || q.contains("serangan"))
        target = "security";
    else if (q.contains("finance") || q.contains("saham") || q.contains("ekonomi"))
        target = "finance";

    m_scene->highlightPathway(target);

    m_txtDetails->setHtml(
        QString("<b style='color:#00ffaa;'>Query:</b> %1<br>"
                "<b style='color:#00ffaa;'>Target:</b> %2<br>"
                "Benang aksion disorot & berdenyut.")
        .arg(q, target.toUpper())
    );
}
