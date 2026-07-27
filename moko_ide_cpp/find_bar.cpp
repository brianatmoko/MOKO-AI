// find_bar.cpp — VSCode-style Find & Replace bar implementation
#include "find_bar.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QTextCursor>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QRegularExpression>
#include <QShortcut>

static const QString FIND_BAR_STYLE = R"(
QWidget#findBar {
    background: #252526;
    border-top: 1px solid #3c3c3c;
    border-bottom: 1px solid #3c3c3c;
}
QLineEdit {
    background: #3c3c3c;
    color: #cccccc;
    border: 1px solid #555;
    border-radius: 2px;
    padding: 4px 8px;
    font-size: 12px;
    selection-background-color: #264f78;
    min-width: 220px;
}
QLineEdit:focus {
    border-color: #007acc;
}
QPushButton#toggleBtn {
    background: transparent;
    border: none;
    color: #666;
    font-size: 11px;
    padding: 2px 4px;
    min-width: 16px;
}
QPushButton#toggleBtn:hover { color: #ccc; }
QPushButton#toggleBtn:checked { color: #007acc; }
QPushButton.optBtn {
    background: transparent;
    border: 1px solid transparent;
    color: #666;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 3px;
    min-width: 22px;
    max-width: 22px;
}
QPushButton.optBtn:hover { color: #ccc; background: #3c3c3c; }
QPushButton.optBtn:checked { color: #007acc; border-color: #007acc; background: rgba(0,122,204,0.12); }
QPushButton.navBtn {
    background: transparent;
    border: none;
    color: #858585;
    font-size: 14px;
    padding: 2px 5px;
    border-radius: 3px;
}
QPushButton.navBtn:hover { color: #cccccc; background: #3c3c3c; }
QPushButton.navBtn:disabled { color: #444; }
QPushButton.actionBtn {
    background: transparent;
    border: 1px solid #555;
    color: #cccccc;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 2px;
}
QPushButton.actionBtn:hover { background: #3c3c3c; border-color: #007acc; color: #fff; }
QPushButton.closeBtn {
    background: transparent;
    border: none;
    color: #666;
    font-size: 14px;
    padding: 0 4px;
}
QPushButton.closeBtn:hover { color: #cccccc; }
QLabel#matchLabel {
    color: #888;
    font-size: 11px;
    min-width: 80px;
    padding: 0 6px;
}
)";

FindBar::FindBar(QWidget* parent) : QWidget(parent) {
    setObjectName("findBar");
    setStyleSheet(FIND_BAR_STYLE);
    setFixedHeight(0);   // hidden by default
    hide();

    QVBoxLayout* vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(8, 4, 8, 4);
    vlay->setSpacing(2);

    // ── FIND ROW ───────────────────────────────────────────────────
    QWidget* findRow = new QWidget(this);
    QHBoxLayout* flay = new QHBoxLayout(findRow);
    flay->setContentsMargins(0, 0, 0, 0);
    flay->setSpacing(4);

    // Toggle replace chevron
    m_btnToggleReplace = new QPushButton("▶", findRow);
    m_btnToggleReplace->setObjectName("toggleBtn");
    m_btnToggleReplace->setCheckable(true);
    m_btnToggleReplace->setToolTip("Toggle Replace");
    m_btnToggleReplace->setFixedSize(16, 22);
    connect(m_btnToggleReplace, &QPushButton::clicked, this, &FindBar::toggleReplaceRow);
    flay->addWidget(m_btnToggleReplace);

    // Find input
    m_findEdit = new QLineEdit(findRow);
    m_findEdit->setPlaceholderText("Find");
    m_findEdit->installEventFilter(this);
    flay->addWidget(m_findEdit, 1);

    // Option toggle buttons
    auto mkOptBtn = [&](const QString& label, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(label, findRow);
        b->setProperty("class", "optBtn");
        b->setCheckable(true);
        b->setToolTip(tip);
        b->setFixedSize(22, 22);
        b->setStyleSheet(
            "QPushButton { background:transparent; border:1px solid transparent; color:#666; "
            "font-size:11px; font-weight:700; border-radius:3px; }"
            "QPushButton:hover { color:#ccc; background:#3c3c3c; }"
            "QPushButton:checked { color:#007acc; border-color:#007acc; background:rgba(0,122,204,0.12); }"
        );
        return b;
    };

    m_btnMatchCase = mkOptBtn("Aa", "Match Case");
    m_btnWholeWord = mkOptBtn("W",  "Match Whole Word");
    m_btnRegex     = mkOptBtn(".*", "Use Regular Expression");
    flay->addWidget(m_btnMatchCase);
    flay->addWidget(m_btnWholeWord);
    flay->addWidget(m_btnRegex);

    // Match count label
    m_matchLabel = new QLabel("", findRow);
    m_matchLabel->setObjectName("matchLabel");
    flay->addWidget(m_matchLabel);

    // Navigation
    auto mkNavBtn = [&](const QString& icon, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(icon, findRow);
        b->setToolTip(tip);
        b->setFixedSize(22, 22);
        b->setStyleSheet(
            "QPushButton { background:transparent; border:none; color:#858585; font-size:14px; border-radius:3px; }"
            "QPushButton:hover { color:#ccc; background:#3c3c3c; }"
            "QPushButton:disabled { color:#444; }"
        );
        return b;
    };

    m_btnPrev = mkNavBtn("↑", "Previous Match (Shift+Enter)");
    m_btnNext = mkNavBtn("↓", "Next Match (Enter)");
    flay->addWidget(m_btnPrev);
    flay->addWidget(m_btnNext);

    // Close
    m_btnClose = new QPushButton("✕", findRow);
    m_btnClose->setToolTip("Close (Escape)");
    m_btnClose->setFixedSize(22, 22);
    m_btnClose->setStyleSheet(
        "QPushButton { background:transparent; border:none; color:#666; font-size:13px; }"
        "QPushButton:hover { color:#ccc; }"
    );
    flay->addWidget(m_btnClose);
    vlay->addWidget(findRow);

    // ── REPLACE ROW ────────────────────────────────────────────────
    m_replaceRow = new QWidget(this);
    m_replaceRow->hide();
    QHBoxLayout* rlay = new QHBoxLayout(m_replaceRow);
    rlay->setContentsMargins(20, 0, 0, 0);
    rlay->setSpacing(4);

    m_replaceEdit = new QLineEdit(m_replaceRow);
    m_replaceEdit->setPlaceholderText("Replace");
    m_replaceEdit->installEventFilter(this);
    rlay->addWidget(m_replaceEdit, 1);

    auto mkActionBtn = [&](const QString& label, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(label, m_replaceRow);
        b->setToolTip(tip);
        b->setStyleSheet(
            "QPushButton { background:transparent; border:1px solid #555; color:#cccccc; "
            "font-size:11px; padding:3px 10px; border-radius:2px; }"
            "QPushButton:hover { background:#3c3c3c; border-color:#007acc; color:#fff; }"
        );
        return b;
    };

    m_btnReplace    = mkActionBtn("Replace",     "Replace (Enter)");
    m_btnReplaceAll = mkActionBtn("Replace All", "Replace All (Ctrl+Alt+Enter)");
    rlay->addWidget(m_btnReplace);
    rlay->addWidget(m_btnReplaceAll);
    rlay->addStretch();
    vlay->addWidget(m_replaceRow);

    // ── Connections ────────────────────────────────────────────────
    connect(m_findEdit,      &QLineEdit::textChanged,  this, &FindBar::onSearchTextChanged);
    connect(m_btnMatchCase,  &QPushButton::clicked,    this, [this]{ onSearchTextChanged(m_findEdit->text()); });
    connect(m_btnWholeWord,  &QPushButton::clicked,    this, [this]{ onSearchTextChanged(m_findEdit->text()); });
    connect(m_btnRegex,      &QPushButton::clicked,    this, [this]{ onSearchTextChanged(m_findEdit->text()); });
    connect(m_btnNext,       &QPushButton::clicked,    this, &FindBar::findNext);
    connect(m_btnPrev,       &QPushButton::clicked,    this, &FindBar::findPrev);
    connect(m_btnReplace,    &QPushButton::clicked,    this, &FindBar::replaceOne);
    connect(m_btnReplaceAll, &QPushButton::clicked,    this, &FindBar::replaceAll);
    connect(m_btnClose,      &QPushButton::clicked,    this, &FindBar::closeBar);

    // Escape closes bar
    new QShortcut(QKeySequence(Qt::Key_Escape), this, SLOT(closeBar()));
}

void FindBar::attachEditor(QPlainTextEdit* editor) {
    m_editor = editor;
}

void FindBar::focusFind() {
    m_findEdit->setFocus();
    m_findEdit->selectAll();
}

void FindBar::show() {
    setFixedHeight(m_replaceVisible ? 68 : 36);
    QWidget::show();
    focusFind();
}

void FindBar::toggleReplaceRow() {
    m_replaceVisible = !m_replaceVisible;
    m_replaceRow->setVisible(m_replaceVisible);
    m_btnToggleReplace->setText(m_replaceVisible ? "▼" : "▶");
    setFixedHeight(m_replaceVisible ? 68 : 36);
}

bool FindBar::eventFilter(QObject* obj, QEvent* ev) {
    if (ev->type() == QEvent::KeyPress) {
        QKeyEvent* ke = static_cast<QKeyEvent*>(ev);
        if (ke->key() == Qt::Key_Return || ke->key() == Qt::Key_Enter) {
            if (ke->modifiers() & Qt::ShiftModifier)
                findPrev();
            else
                findNext();
            return true;
        }
        if (ke->key() == Qt::Key_Escape) {
            closeBar();
            return true;
        }
    }
    return QWidget::eventFilter(obj, ev);
}

QTextDocument::FindFlags FindBar::buildFlags() const {
    QTextDocument::FindFlags f;
    if (m_btnMatchCase->isChecked()) f |= QTextDocument::FindCaseSensitively;
    if (m_btnWholeWord->isChecked()) f |= QTextDocument::FindWholeWords;
    return f;
}

void FindBar::highlightAll(const QString& term) {
    if (!m_editor || term.isEmpty()) {
        clearHighlights();
        m_matchCount = 0;
        updateMatchLabel();
        return;
    }

    QList<QTextEdit::ExtraSelection> extras;
    QTextDocument* doc = m_editor->document();
    QTextCharFormat highlightFmt;
    highlightFmt.setBackground(QColor("#613214"));   // orange-ish, like VSCode
    highlightFmt.setForeground(QColor("#ffffff"));

    QTextCharFormat currentFmt;
    currentFmt.setBackground(QColor("#9e6a03"));     // bright current match
    currentFmt.setForeground(QColor("#ffffff"));

    m_matchCount = 0;

    if (m_btnRegex->isChecked()) {
        QRegularExpression re(term,
            m_btnMatchCase->isChecked()
                ? QRegularExpression::NoPatternOption
                : QRegularExpression::CaseInsensitiveOption);
        if (!re.isValid()) {
            clearHighlights();
            m_matchLabel->setText("Invalid regex");
            m_matchLabel->setStyleSheet("color:#f48771; font-size:11px; padding:0 6px;");
            return;
        }
        QRegularExpressionMatchIterator it = re.globalMatch(doc->toPlainText());
        while (it.hasNext()) {
            auto match = it.next();
            QTextCursor c(doc);
            c.setPosition(match.capturedStart());
            c.setPosition(match.capturedEnd(), QTextCursor::KeepAnchor);
            QTextEdit::ExtraSelection sel;
            sel.cursor = c;
            sel.format = highlightFmt;
            extras.append(sel);
            m_matchCount++;
        }
    } else {
        QTextDocument::FindFlags flags = buildFlags();
        QTextCursor cursor = doc->find(term, 0, flags);
        while (!cursor.isNull()) {
            QTextEdit::ExtraSelection sel;
            sel.cursor = cursor;
            sel.format = highlightFmt;
            extras.append(sel);
            m_matchCount++;
            cursor = doc->find(term, cursor, flags);
        }
    }

    m_editor->setExtraSelections(extras);
    updateMatchLabel();
}

void FindBar::clearHighlights() {
    if (m_editor)
        m_editor->setExtraSelections({});
}

void FindBar::updateMatchLabel() {
    if (m_matchCount == 0 && !m_findEdit->text().isEmpty()) {
        m_matchLabel->setText("No results");
        m_matchLabel->setStyleSheet("color:#f48771; font-size:11px; padding:0 6px;");
        m_findEdit->setStyleSheet(
            "QLineEdit { background:#5a1d1d; color:#f48771; border:1px solid #f48771; "
            "border-radius:2px; padding:4px 8px; font-size:12px; }"
        );
    } else if (m_matchCount > 0) {
        m_matchLabel->setText(QString("%1 of %2").arg(m_currentMatch).arg(m_matchCount));
        m_matchLabel->setStyleSheet("color:#888; font-size:11px; padding:0 6px;");
        m_findEdit->setStyleSheet(
            "QLineEdit { background:#3c3c3c; color:#cccccc; border:1px solid #007acc; "
            "border-radius:2px; padding:4px 8px; font-size:12px; }"
        );
    } else {
        m_matchLabel->clear();
        m_matchLabel->setStyleSheet("");
        m_findEdit->setStyleSheet(
            "QLineEdit { background:#3c3c3c; color:#cccccc; border:1px solid #555; "
            "border-radius:2px; padding:4px 8px; font-size:12px; }"
        );
    }
}

void FindBar::onSearchTextChanged(const QString& text) {
    m_currentMatch = 0;
    highlightAll(text);
    if (!text.isEmpty() && m_matchCount > 0) {
        findNext();
    }
}

void FindBar::doFindFrom(bool forward, QTextCursor startCursor) {
    if (!m_editor || m_findEdit->text().isEmpty()) return;
    QString term = m_findEdit->text();
    QTextDocument* doc = m_editor->document();
    QTextDocument::FindFlags flags = buildFlags();
    if (!forward) flags |= QTextDocument::FindBackward;

    QTextCursor found;
    if (m_btnRegex->isChecked()) {
        QRegularExpression re(term,
            m_btnMatchCase->isChecked()
                ? QRegularExpression::NoPatternOption
                : QRegularExpression::CaseInsensitiveOption);
        found = doc->find(re, startCursor, flags);
        if (found.isNull()) {
            // Wrap around
            QTextCursor wrap(doc);
            wrap.movePosition(forward ? QTextCursor::Start : QTextCursor::End);
            found = doc->find(re, wrap, flags);
        }
    } else {
        found = doc->find(term, startCursor, flags);
        if (found.isNull()) {
            QTextCursor wrap(doc);
            wrap.movePosition(forward ? QTextCursor::Start : QTextCursor::End);
            found = doc->find(term, wrap, flags);
        }
    }

    if (!found.isNull()) {
        m_editor->setTextCursor(found);
        m_editor->ensureCursorVisible();
        // Update current match index
        if (forward) {
            m_currentMatch = (m_currentMatch % m_matchCount) + 1;
        } else {
            m_currentMatch = m_currentMatch <= 1 ? m_matchCount : m_currentMatch - 1;
        }
        updateMatchLabel();
    }
}

void FindBar::findNext() {
    if (!m_editor) return;
    QTextCursor cur = m_editor->textCursor();
    doFindFrom(true, cur);
}

void FindBar::findPrev() {
    if (!m_editor) return;
    QTextCursor cur = m_editor->textCursor();
    doFindFrom(false, cur);
}

void FindBar::replaceOne() {
    if (!m_editor) return;
    QTextCursor cur = m_editor->textCursor();
    if (cur.hasSelection()) {
        cur.insertText(m_replaceEdit->text());
    }
    findNext();
    highlightAll(m_findEdit->text());
}

void FindBar::replaceAll() {
    if (!m_editor || m_findEdit->text().isEmpty()) return;
    QString term    = m_findEdit->text();
    QString replace = m_replaceEdit->text();
    QTextDocument* doc = m_editor->document();
    QTextDocument::FindFlags flags = buildFlags();

    QTextCursor cursor(doc);
    cursor.beginEditBlock();
    int count = 0;
    if (m_btnRegex->isChecked()) {
        QRegularExpression re(term,
            m_btnMatchCase->isChecked()
                ? QRegularExpression::NoPatternOption
                : QRegularExpression::CaseInsensitiveOption);
        while (true) {
            QTextCursor found = doc->find(re, cursor, flags);
            if (found.isNull()) break;
            found.insertText(replace);
            cursor = found;
            count++;
        }
    } else {
        while (true) {
            QTextCursor found = doc->find(term, cursor, flags);
            if (found.isNull()) break;
            found.insertText(replace);
            cursor = found;
            count++;
        }
    }
    cursor.endEditBlock();
    clearHighlights();
    m_matchCount = 0;
    m_currentMatch = 0;
    m_matchLabel->setText(QString("%1 replacements").arg(count));
    m_matchLabel->setStyleSheet("color:#89d185; font-size:11px; padding:0 6px;");
}

void FindBar::closeBar() {
    clearHighlights();
    hide();
    setFixedHeight(0);
    if (m_editor) m_editor->setFocus();
}
