// syntax_highlighter.h — Multi-language syntax highlighter (VSCode Dark+ colors)
#ifndef SYNTAX_HIGHLIGHTER_H
#define SYNTAX_HIGHLIGHTER_H

#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QRegularExpression>
#include <QVector>

class SyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
public:
    explicit SyntaxHighlighter(QTextDocument* parent, const QString& language = "python");

protected:
    void highlightBlock(const QString& text) override;

private:
    struct Rule {
        QRegularExpression pattern;
        QTextCharFormat    format;
    };
    QVector<Rule> m_rules;

    // Python multi-line string state (state 1 = ''', state 2 = """)
    QRegularExpression m_tripleQuoteSingle;
    QRegularExpression m_tripleQuoteDouble;
    QTextCharFormat    m_stringFormat;

    // C/C++/JS/CSS/HTML block comment state (state 10)
    QRegularExpression m_multilineCommentStart;
    QRegularExpression m_multilineCommentEnd;
};

#endif // SYNTAX_HIGHLIGHTER_H
