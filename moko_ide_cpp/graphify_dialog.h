#ifndef GRAPHIFY_DIALOG_H
#define GRAPHIFY_DIALOG_H

#include <QDialog>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QTimer>
#include <QList>
#include <QMap>
#include <QJsonObject>
#include <QVector2D>

class QLabel;
class QPushButton;
class QLineEdit;
class QTextBrowser;
class QFrame;

// ──────────────────────────────────────────────────────────────────────────────
// NeuronNodeItem — Represents a single neuron cell body (Domain, Bucket, or Memory)
//   Extremely clean: tiny glowing dots, no heavy circles or text unless hovered.
// ──────────────────────────────────────────────────────────────────────────────
class NeuronNodeItem : public QGraphicsItem {
public:
    enum NodeType { DomainCell, BucketCell, MemoryCell };

    NeuronNodeItem(const QString& id, const QString& label, NodeType type, const QColor& color);

    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;

    QString    m_id;
    QString    m_label;
    NodeType   m_type;
    QColor     m_color;
    QPointF    m_pos;
    bool       m_hovered = false;

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent* e) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* e) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* e) override;
};

// ──────────────────────────────────────────────────────────────────────────────
// NeuronEdgeItem — A super-thin, organic neural fiber (axon/dendrite)
// ──────────────────────────────────────────────────────────────────────────────
class NeuronEdgeItem : public QGraphicsItem {
public:
    NeuronEdgeItem(NeuronNodeItem* src, NeuronNodeItem* dst, const QColor& color, double weight = 1.0);

    QRectF boundingRect() const override;
    void   paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;

    void setPulsePhase(double phase) { m_pulsePhase = phase; }

    NeuronNodeItem* m_src;
    NeuronNodeItem* m_dst;
    QColor          m_color;
    double          m_weight;
    bool            m_highlighted = false;
    double          m_pulsePhase  = 0.0;
    double          m_phaseOffset = 0.0;
};

// ──────────────────────────────────────────────────────────────────────────────
// NeuronScene — The custom scene managing the neural network
// ──────────────────────────────────────────────────────────────────────────────
class NeuronScene : public QGraphicsScene {
    Q_OBJECT
public:
    explicit NeuronScene(QObject* parent = nullptr);

    void addNeuron(NeuronNodeItem* node);
    void addFiber(NeuronEdgeItem* edge);
    void clearAll();
    void highlightPathway(const QString& nodeId);

    double pulsePhase() const { return m_pulsePhase; }

public slots:
    void animateNetwork();

private:
    QList<NeuronNodeItem*> m_nodes;
    QList<NeuronEdgeItem*> m_edges;
    double                 m_pulsePhase = 0.0;
};

// ──────────────────────────────────────────────────────────────────────────────
// MokoGraphifyDialog — Main Window Dialog
// ──────────────────────────────────────────────────────────────────────────────
class MokoGraphifyDialog : public QDialog {
    Q_OBJECT
public:
    explicit MokoGraphifyDialog(QWidget* parent = nullptr);
    ~MokoGraphifyDialog();

private slots:
    void refreshData();
    void runSearch();
    void onNodeSelected(QGraphicsItem* item);

private:
    void buildUi();
    void scanOmniStore();

    QString        m_omniRoot;
    QTimer*        m_animTimer;
    NeuronScene*   m_scene;
    QGraphicsView* m_view;

    // UI Widgets
    QLabel*       m_lblTotalMemories;
    QLabel*       m_lblActiveDomains;
    QLabel*       m_lblActiveBuckets;
    QTextBrowser* m_txtStats;
    QTextBrowser* m_txtDetails;
    QLineEdit*    m_txtSearch;
    QPushButton*  m_btnSearch;

    struct MemoryRecord {
        QString contentHash;
        QString text;
        QString source;
        QString domain;
        QString folder;
        double  valence;
        double  arousal;
        int     consolidatedCount;
        qint64  timestamp;
    };
    QList<MemoryRecord> m_records;
};

#endif // GRAPHIFY_DIALOG_H
