// code_editor.h — VSCode-like QPlainTextEdit with Gutter, Minimap, Bracket Matching, and Auto-Close
#ifndef CODE_EDITOR_H
#define CODE_EDITOR_H

#include <QPlainTextEdit>
#include <QWidget>
#include <QPainter>
#include <QTextBlock>
#include <QPaintEvent>

class LineNumberArea;
class MinimapArea;

class CodeEditor : public QPlainTextEdit {
    Q_OBJECT
public:
    explicit CodeEditor(QWidget* parent = nullptr);

    // Gutter & Minimap paint delegates
    void lineNumberAreaPaintEvent(QPaintEvent* event);
    void minimapAreaPaintEvent(QPaintEvent* event);

    int  lineNumberAreaWidth() const;
    int  minimapAreaWidth() const;

    bool isModified() const { return m_modified; }
    void setModified(bool v);
    void setMinimapVisible(bool visible);

signals:
    void modifiedStateChanged(bool modified);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void updateLineNumberAreaWidth(int newBlockCount);
    void highlightCurrentLineAndBrackets();
    void updateEditorViews(const QRect& rect, int dy);
    void onContentsChanged();

private:
    LineNumberArea* m_lineNumberArea;
    MinimapArea*    m_minimapArea;
    bool            m_modified = false;
    bool            m_minimapVisible = true;

    // Helper functions for brackets and guides
    void matchBrackets();
    void drawIndentGuides(QPainter& painter, QTextBlock block, int top, int bottom);
};

// ── LineNumberArea Gutter ──────────────────────────────────────────────────────
class LineNumberArea : public QWidget {
public:
    explicit LineNumberArea(CodeEditor* editor) : QWidget(editor), m_editor(editor) {}

    QSize sizeHint() const override {
        return { m_editor->lineNumberAreaWidth(), 0 };
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        m_editor->lineNumberAreaPaintEvent(event);
    }

private:
    CodeEditor* m_editor;
};

// ── MinimapArea Gutter ────────────────────────────────────────────────────────
class MinimapArea : public QWidget {
public:
    explicit MinimapArea(CodeEditor* editor) : QWidget(editor), m_editor(editor) {}

    QSize sizeHint() const override {
        return { m_editor->minimapAreaWidth(), 0 };
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        m_editor->minimapAreaPaintEvent(event);
    }

private:
    CodeEditor* m_editor;
};

#endif // CODE_EDITOR_H
