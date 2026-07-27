// find_bar.h — VSCode-style inline Find & Replace bar
#ifndef FIND_BAR_H
#define FIND_BAR_H

#include <QWidget>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QKeyEvent>
#include <QTextDocument>

class FindBar : public QWidget {
    Q_OBJECT
public:
    explicit FindBar(QWidget* parent = nullptr);

    void attachEditor(QPlainTextEdit* editor);
    void focusFind();   // Focus the find input
    void show();

public slots:
    void findNext();
    void findPrev();
    void replaceOne();
    void replaceAll();
    void closeBar();

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;

private slots:
    void onSearchTextChanged(const QString& text);
    void toggleReplaceRow();

private:
    // Find row
    QPushButton*  m_btnToggleReplace;
    QLineEdit*    m_findEdit;
    QPushButton*  m_btnMatchCase;
    QPushButton*  m_btnWholeWord;
    QPushButton*  m_btnRegex;
    QLabel*       m_matchLabel;
    QPushButton*  m_btnPrev;
    QPushButton*  m_btnNext;
    QPushButton*  m_btnClose;

    // Replace row
    QWidget*      m_replaceRow;
    QLineEdit*    m_replaceEdit;
    QPushButton*  m_btnReplace;
    QPushButton*  m_btnReplaceAll;

    QPlainTextEdit* m_editor = nullptr;
    bool m_replaceVisible = false;
    int  m_matchCount     = 0;
    int  m_currentMatch   = 0;

    QTextDocument::FindFlags buildFlags() const;
    void highlightAll(const QString& term);
    void clearHighlights();
    void updateMatchLabel();
    void doFindFrom(bool forward, QTextCursor startCursor);
};

#endif // FIND_BAR_H
