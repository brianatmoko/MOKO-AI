// code_editor.cpp — VSCode-like editor with Minimap, Auto-Brackets, Bracket Matching, and Indent Guides
#include "code_editor.h"
#include <QTextBlock>
#include <QAbstractTextDocumentLayout>
#include <QScrollBar>
#include <QFont>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QTextCharFormat>
#include <QTextBlockFormat>

CodeEditor::CodeEditor(QWidget* parent)
    : QPlainTextEdit(parent)
    , m_lineNumberArea(new LineNumberArea(this))
    , m_minimapArea(new MinimapArea(this))
    , m_modified(false)
    , m_minimapVisible(true)
{
    // ── Monospace font & Line Height ─────────────────────────────
    QFont font;
    font.setFamily("JetBrains Mono");
    font.setStyleHint(QFont::Monospace);
    font.setFixedPitch(true);
    font.setPointSize(11);
    setFont(font);

    // Set custom tab stop width (4 spaces)
    setTabStopDistance(QFontMetrics(font).horizontalAdvance(' ') * 4);
    setLineWrapMode(QPlainTextEdit::NoWrap);

    // Visual Cursor settings
    setCursorWidth(2); // Thick, modern cursor like VSCode

    // Apply 1.25x line height spacing to all text blocks via document layout
    QTextBlockFormat format;
    format.setLineHeight(125, QTextBlockFormat::ProportionalHeight);
    QTextCursor cursor = textCursor();
    cursor.select(QTextCursor::Document);
    cursor.mergeBlockFormat(format);

    // ── Signals ──────────────────────────────────────────────────
    connect(this, &CodeEditor::blockCountChanged, this, &CodeEditor::updateLineNumberAreaWidth);
    connect(this, &CodeEditor::updateRequest, this, &CodeEditor::updateEditorViews);
    connect(this, &CodeEditor::cursorPositionChanged, this, &CodeEditor::highlightCurrentLineAndBrackets);
    connect(document(), &QTextDocument::contentsChanged, this, &CodeEditor::onContentsChanged);

    updateLineNumberAreaWidth(0);
    highlightCurrentLineAndBrackets();
}

int CodeEditor::lineNumberAreaWidth() const {
    int digits = 1;
    int max = qMax(1, blockCount());
    while (max >= 10) { max /= 10; ++digits; }
    int space = 8 + fontMetrics().horizontalAdvance('9') * digits + 12;
    return space;
}

int CodeEditor::minimapAreaWidth() const {
    return m_minimapVisible ? 60 : 0;
}

void CodeEditor::setMinimapVisible(bool visible) {
    m_minimapVisible = visible;
    m_minimapArea->setVisible(visible);
    updateLineNumberAreaWidth(0);
}

void CodeEditor::updateLineNumberAreaWidth(int) {
    setViewportMargins(lineNumberAreaWidth(), 0, minimapAreaWidth(), 0);
}

void CodeEditor::updateEditorViews(const QRect& rect, int dy) {
    if (dy) {
        m_lineNumberArea->scroll(0, dy);
        if (m_minimapVisible) m_minimapArea->scroll(0, dy);
    } else {
        m_lineNumberArea->update(0, rect.y(), m_lineNumberArea->width(), rect.height());
        if (m_minimapVisible) m_minimapArea->update(0, rect.y(), m_minimapArea->width(), rect.height());
    }

    if (rect.contains(viewport()->rect()))
        updateLineNumberAreaWidth(0);
}

void CodeEditor::resizeEvent(QResizeEvent* e) {
    QPlainTextEdit::resizeEvent(e);
    QRect cr = contentsRect();
    m_lineNumberArea->setGeometry(QRect(cr.left(), cr.top(), lineNumberAreaWidth(), cr.height()));
    if (m_minimapVisible) {
        m_minimapArea->setGeometry(QRect(cr.right() - minimapAreaWidth(), cr.top(), minimapAreaWidth(), cr.height()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bracket Matching & Current Line Highlight
// ─────────────────────────────────────────────────────────────────────────────
void CodeEditor::highlightCurrentLineAndBrackets() {
    QList<QTextEdit::ExtraSelection> selections;

    // 1. Highlight current active line (subtle background bar)
    if (!isReadOnly()) {
        QTextEdit::ExtraSelection sel;
        sel.format.setBackground(QColor("#121212"));
        sel.format.setProperty(QTextFormat::FullWidthSelection, true);
        sel.cursor = textCursor();
        sel.cursor.clearSelection();
        selections.append(sel);
    }

    // 2. Bracket Matching
    QTextCursor cur = textCursor();
    int pos = cur.position();
    QString text = document()->toPlainText();
    
    auto matchChar = [&](int p, QChar open, QChar close, int dir) {
        int depth = 1;
        int i = p + dir;
        while (i >= 0 && i < text.length()) {
            if (text[i] == open) depth += dir;
            else if (text[i] == close) depth -= dir;
            if (depth == 0) {
                QTextEdit::ExtraSelection sel1, sel2;
                QTextCharFormat fmt;
                fmt.setBackground(QColor("#3d241d"));
                fmt.setForeground(QColor("#ffffff"));
                fmt.setFontWeight(QFont::Bold);

                sel1.format = fmt;
                sel1.cursor = textCursor();
                sel1.cursor.setPosition(p);
                sel1.cursor.setPosition(p + 1, QTextCursor::KeepAnchor);

                sel2.format = fmt;
                sel2.cursor = textCursor();
                sel2.cursor.setPosition(i);
                sel2.cursor.setPosition(i + 1, QTextCursor::KeepAnchor);

                selections.append(sel1);
                selections.append(sel2);
                break;
            }
            i += dir;
        }
    };

    if (pos < text.length()) {
        QChar c = text[pos];
        if (c == '{') matchChar(pos, '{', '}', 1);
        else if (c == '[') matchChar(pos, '[', ']', 1);
        else if (c == '(') matchChar(pos, '(', ')', 1);
        else if (c == '}') matchChar(pos, '}', '{', -1);
        else if (c == ']') matchChar(pos, ']', '[', -1);
        else if (c == ')') matchChar(pos, ')', '(', -1);
    }

    setExtraSelections(selections);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gutter Paint Event (Line Numbers)
// ─────────────────────────────────────────────────────────────────────────────
void CodeEditor::lineNumberAreaPaintEvent(QPaintEvent* event) {
    QPainter painter(m_lineNumberArea);
    painter.fillRect(event->rect(), QColor("#0d0d0d"));

    // Subtle right border
    painter.setPen(QColor("#1a1a1a"));
    painter.drawLine(m_lineNumberArea->width() - 1, event->rect().top(),
                     m_lineNumberArea->width() - 1, event->rect().bottom());

    QTextBlock block = firstVisibleBlock();
    int blockNumber = block.blockNumber();
    int top = qRound(blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + qRound(blockBoundingRect(block).height());
    int curLine = textCursor().blockNumber();

    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            bool isCurrent = (blockNumber == curLine);
            painter.setPen(isCurrent ? QColor("#ff6b35") : QColor("#444444"));

            if (isCurrent) {
                QFont f = painter.font();
                f.setBold(true);
                painter.setFont(f);
            }

            int blockHeight = qRound(blockBoundingRect(block).height());
            painter.drawText(0, top, m_lineNumberArea->width() - 10, blockHeight,
                             Qt::AlignRight | Qt::AlignVCenter, QString::number(blockNumber + 1));

            if (isCurrent) {
                QFont f = painter.font();
                f.setBold(false);
                painter.setFont(f);
            }
        }
        block = block.next();
        top = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
        ++blockNumber;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimap Paint Event (Visual summary of source code)
// ─────────────────────────────────────────────────────────────────────────────
void CodeEditor::minimapAreaPaintEvent(QPaintEvent* event) {
    QPainter painter(m_minimapArea);
    painter.fillRect(event->rect(), QColor("#0d0d0d")); // Gutter color matching VSCode

    // Draw scroll indicator outline (active viewport area on minimap)
    int viewTop = verticalScrollBar()->value();
    int viewMax = verticalScrollBar()->maximum();
    int viewVisible = viewport()->height();

    int scrollAreaHeight = m_minimapArea->height();
    int indicatorY = (viewTop * (scrollAreaHeight - 40)) / qMax(1, viewMax + viewVisible);
    int indicatorH = (viewVisible * scrollAreaHeight) / qMax(1.0, document()->size().height() + viewVisible);

    painter.fillRect(0, indicatorY, m_minimapArea->width(), qMax(30, indicatorH), QColor(255, 255, 255, 12));

    // Render text with 3px size font
    QFont miniFont("Courier", 3);
    painter.setFont(miniFont);
    painter.setPen(QColor("#4a4a4a"));

    QTextBlock block = firstVisibleBlock();
    int top = 0;
    while (block.isValid() && top < m_minimapArea->height()) {
        QString txt = block.text();
        if (!txt.trimmed().isEmpty()) {
            // Draw up to 24 chars to avoid overflowing
            painter.drawText(4, top, m_minimapArea->width() - 8, 5, Qt::AlignLeft, txt.left(24));
        }
        top += 5;
        block = block.next();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Indent Guides Drawer
// ─────────────────────────────────────────────────────────────────────────────
void CodeEditor::drawIndentGuides(QPainter& painter, QTextBlock block, int top, int bottom) {
    QString text = block.text();
    int spaces = 0;
    for (QChar c : text) {
        if (c == ' ') spaces++;
        else if (c == '\t') spaces += 4;
        else break;
    }

    int guidesCount = spaces / 4;
    if (guidesCount <= 0) return;

    painter.setPen(QPen(QColor("#2f2f2f"), 1, Qt::SolidLine));
    int charWidth = fontMetrics().horizontalAdvance(' ');
    int leftOffset = lineNumberAreaWidth() + viewportMargins().left() + 4; // Margin spacer

    for (int i = 1; i <= guidesCount; ++i) {
        int x = leftOffset + (i * 4 * charWidth);
        painter.drawLine(x, top, x, bottom);
    }
}

void CodeEditor::paintEvent(QPaintEvent* event) {
    QPlainTextEdit::paintEvent(event);

    // Draw Indent Guides after base painter finishes
    QPainter painter(viewport());
    QTextBlock block = firstVisibleBlock();
    int top = qRound(blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + qRound(blockBoundingRect(block).height());

    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            drawIndentGuides(painter, block, top, bottom);
        }
        block = block.next();
        top = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Bracket & Quotes Close
// ─────────────────────────────────────────────────────────────────────────────
void CodeEditor::keyPressEvent(QKeyEvent* event) {
    QTextCursor cursor = textCursor();
    int pos = cursor.position();
    QString text = document()->toPlainText();
    QChar nextChar = (pos < text.length()) ? text[pos] : QChar();

    auto insertPair = [&](const QString& open, const QString& close) {
        cursor.beginEditBlock();
        cursor.insertText(open + close);
        cursor.movePosition(QTextCursor::Left, QTextCursor::MoveAnchor, close.length());
        setTextCursor(cursor);
        cursor.endEditBlock();
    };

    auto skipOrInsert = [&](const QString& closingChar) {
        if (nextChar == closingChar[0]) {
            cursor.movePosition(QTextCursor::Right);
            setTextCursor(cursor);
        } else {
            QPlainTextEdit::keyPressEvent(event);
        }
    };

    switch (event->key()) {
    case Qt::Key_ParenLeft:
        insertPair("(", ")");
        break;
    case Qt::Key_BracketLeft:
        insertPair("[", "]");
        break;
    case Qt::Key_BraceLeft:
        insertPair("{", "}");
        break;
    case Qt::Key_QuoteDbl:
        if (nextChar == '"') {
            cursor.movePosition(QTextCursor::Right);
            setTextCursor(cursor);
        } else {
            insertPair("\"", "\"");
        }
        break;
    case Qt::Key_Apostrophe:
        if (nextChar == '\'') {
            cursor.movePosition(QTextCursor::Right);
            setTextCursor(cursor);
        } else {
            insertPair("'", "'");
        }
        break;
    case Qt::Key_ParenRight:
        skipOrInsert(")");
        break;
    case Qt::Key_BracketRight:
        skipOrInsert("]");
        break;
    case Qt::Key_BraceRight:
        skipOrInsert("}");
        break;
    default:
        QPlainTextEdit::keyPressEvent(event);
        break;
    }
}

void CodeEditor::onContentsChanged() {
    if (!m_modified) {
        m_modified = true;
        emit modifiedStateChanged(true);
    }
}

void CodeEditor::setModified(bool v) {
    m_modified = v;
    emit modifiedStateChanged(v);
}
