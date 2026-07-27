// syntax_highlighter.cpp — VSCode Dark+ token colors for multiple languages
#include "syntax_highlighter.h"

SyntaxHighlighter::SyntaxHighlighter(QTextDocument* parent, const QString& language)
    : QSyntaxHighlighter(parent)
{
    // ── Token color palette (VSCode Dark+) ────────────────────────────────────
    QTextCharFormat kwFmt;          // keyword: blue
    kwFmt.setForeground(QColor("#569cd6"));
    kwFmt.setFontWeight(QFont::Bold);

    QTextCharFormat kw2Fmt;         // control flow / storage: purple
    kw2Fmt.setForeground(QColor("#c586c0"));
    kw2Fmt.setFontWeight(QFont::Bold);

    QTextCharFormat typeFmt;        // type names / builtins: teal
    typeFmt.setForeground(QColor("#4ec9b0"));

    QTextCharFormat defFmt;         // function / class definitions: yellow
    defFmt.setForeground(QColor("#dcdcaa"));

    QTextCharFormat callFmt;        // function calls: yellow (lighter)
    callFmt.setForeground(QColor("#dcdcaa"));

    QTextCharFormat strFmt;         // strings: orange
    strFmt.setForeground(QColor("#ce9178"));

    QTextCharFormat numFmt;         // numbers: green-ish
    numFmt.setForeground(QColor("#b5cea8"));

    QTextCharFormat cmtFmt;         // comments: green italic
    cmtFmt.setForeground(QColor("#6a9955"));
    cmtFmt.setFontItalic(true);

    QTextCharFormat selfFmt;        // self/this: light blue
    selfFmt.setForeground(QColor("#9cdcfe"));

    QTextCharFormat decorFmt;       // decorators / annotations: yellow-green
    decorFmt.setForeground(QColor("#c586c0"));

    QTextCharFormat opFmt;          // operators
    opFmt.setForeground(QColor("#d4d4d4"));

    QTextCharFormat attrFmt;        // attributes / properties
    attrFmt.setForeground(QColor("#9cdcfe"));

    QTextCharFormat macroFmt;       // preprocessor / macros
    macroFmt.setForeground(QColor("#c586c0"));

    QTextCharFormat tagFmt;         // HTML/XML tags
    tagFmt.setForeground(QColor("#569cd6"));

    QTextCharFormat attrNameFmt;    // HTML attribute names
    attrNameFmt.setForeground(QColor("#9cdcfe"));

    QTextCharFormat selectorFmt;    // CSS selectors
    selectorFmt.setForeground(QColor("#d7ba7d"));

    QTextCharFormat propFmt;        // CSS properties
    propFmt.setForeground(QColor("#9cdcfe"));

    m_stringFormat = strFmt;
    m_tripleQuoteSingle = QRegularExpression("'''");
    m_tripleQuoteDouble = QRegularExpression("\"\"\"");

    // ─────────────────────────────────────────────────────────────────────────
    // PYTHON
    // ─────────────────────────────────────────────────────────────────────────
    if (language == "python") {
        const QStringList keywords = {
            "\\bFalse\\b","\\bNone\\b","\\bTrue\\b","\\band\\b","\\bas\\b","\\bassert\\b",
            "\\basync\\b","\\bawait\\b","\\bbreak\\b","\\bclass\\b","\\bcontinue\\b",
            "\\bdef\\b","\\bdel\\b","\\belif\\b","\\belse\\b","\\bexcept\\b",
            "\\bfinally\\b","\\bfor\\b","\\bfrom\\b","\\bglobal\\b","\\bif\\b",
            "\\bimport\\b","\\bin\\b","\\bis\\b","\\blambda\\b","\\bnonlocal\\b",
            "\\bnot\\b","\\bor\\b","\\bpass\\b","\\braise\\b","\\breturn\\b",
            "\\btry\\b","\\bwhile\\b","\\bwith\\b","\\byield\\b","\\bmatch\\b","\\bcase\\b"
        };
        for (const auto& kw : keywords)
            m_rules.append({QRegularExpression(kw), kwFmt});

        const QStringList builtins = {
            "\\bprint\\b","\\blen\\b","\\brange\\b","\\btype\\b","\\bstr\\b","\\bint\\b",
            "\\bfloat\\b","\\blist\\b","\\bdict\\b","\\bset\\b","\\btuple\\b",
            "\\bbool\\b","\\bopen\\b","\\bsuper\\b","\\bstaticmethod\\b","\\bclassmethod\\b",
            "\\bproperty\\b","\\bisinstance\\b","\\bhasattr\\b","\\bgetattr\\b","\\bsetattr\\b",
            "\\bEnumerate\\b","\\bmap\\b","\\bfilter\\b","\\bzip\\b","\\bsorted\\b",
            "\\breversed\\b","\\bany\\b","\\ball\\b","\\bmin\\b","\\bmax\\b","\\bsum\\b",
            "\\babs\\b","\\bchr\\b","\\bord\\b","\\bhex\\b","\\boct\\b","\\bbin\\b",
            "\\bException\\b","\\bValueError\\b","\\bTypeError\\b","\\bRuntimeError\\b",
            "\\bKeyError\\b","\\bIndexError\\b","\\bAttributeError\\b","\\bNotImplementedError\\b",
            "\\bOSError\\b","\\bIOError\\b","\\bStopIteration\\b","\\bGeneratorExit\\b"
        };
        for (const auto& b : builtins)
            m_rules.append({QRegularExpression(b), typeFmt});

        // def / class names
        m_rules.append({QRegularExpression("(?<=\\bdef\\s)\\w+"), defFmt});
        m_rules.append({QRegularExpression("(?<=\\bclass\\s)\\w+"), defFmt});

        // function calls
        m_rules.append({QRegularExpression("\\b(\\w+)(?=\\s*\\()"), callFmt});

        // self, cls
        m_rules.append({QRegularExpression("\\bself\\b|\\bcls\\b"), selfFmt});

        // decorators
        m_rules.append({QRegularExpression("@[\\w.]+"), decorFmt});

        // f-strings prefix (raw/f/b)
        m_rules.append({QRegularExpression("\\b[fFrRbBuU]+(?=['\"])"), decorFmt});

        // numbers
        m_rules.append({QRegularExpression("\\b\\d+\\.?\\d*([eE][+-]?\\d+)?[jJ]?\\b|0x[0-9A-Fa-f]+|0o[0-7]+|0b[01]+"), numFmt});

        // strings
        m_rules.append({QRegularExpression("\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"|'[^'\\\\]*(\\\\.[^'\\\\]*)*'"), strFmt});

        // comments
        m_rules.append({QRegularExpression("#[^\n]*"), cmtFmt});

    // ─────────────────────────────────────────────────────────────────────────
    // C / C++
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "cpp" || language == "c") {
        const QStringList keywords = {
            "\\bauto\\b","\\bbool\\b","\\bbreak\\b","\\bcase\\b","\\bchar\\b",
            "\\bclass\\b","\\bconst\\b","\\bconstexpr\\b","\\bconsteval\\b",
            "\\bcontinue\\b","\\bdefault\\b","\\bdelete\\b","\\bdo\\b","\\bdouble\\b",
            "\\belse\\b","\\benum\\b","\\bexplicit\\b","\\bextern\\b","\\bfloat\\b",
            "\\bfor\\b","\\bfriend\\b","\\bgoto\\b","\\bif\\b","\\binline\\b",
            "\\bint\\b","\\blong\\b","\\bmutable\\b","\\bnamespace\\b","\\bnew\\b",
            "\\bnoexcept\\b","\\bnullptr\\b","\\boperator\\b","\\boverride\\b",
            "\\bprivate\\b","\\bprotected\\b","\\bpublic\\b","\\bregister\\b",
            "\\breturn\\b","\\bshort\\b","\\bsigned\\b","\\bsizeof\\b",
            "\\bstatic\\b","\\bstatic_assert\\b","\\bstatic_cast\\b",
            "\\bdynamic_cast\\b","\\breinterpret_cast\\b","\\bconst_cast\\b",
            "\\bstruct\\b","\\bswitch\\b","\\btemplate\\b","\\bthis\\b",
            "\\bthrow\\b","\\btry\\b","\\btypedef\\b","\\btypename\\b",
            "\\bunion\\b","\\bunsigned\\b","\\busing\\b","\\bvirtual\\b",
            "\\bvoid\\b","\\bvolatile\\b","\\bwhile\\b",
            "\\btrue\\b","\\bfalse\\b"
        };
        for (const auto& kw : keywords)
            m_rules.append({QRegularExpression(kw), kwFmt});

        // STL types
        const QStringList types = {
            "\\bstd::\\w+","\\bstring\\b","\\bvector\\b","\\bmap\\b","\\bset\\b",
            "\\bunordered_map\\b","\\bunordered_set\\b","\\blist\\b","\\bdeque\\b",
            "\\barray\\b","\\bpair\\b","\\btuple\\b","\\boptional\\b","\\bvariant\\b",
            "\\bshared_ptr\\b","\\bunique_ptr\\b","\\bweak_ptr\\b","\\bfunction\\b",
            "\\bsize_t\\b","\\bint8_t\\b","\\bint16_t\\b","\\bint32_t\\b","\\bint64_t\\b",
            "\\buint8_t\\b","\\buint16_t\\b","\\buint32_t\\b","\\buint64_t\\b",
            "\\bQString\\b","\\bQVector\\b","\\bQList\\b","\\bQMap\\b","\\bQHash\\b",
            "\\bQWidget\\b","\\bQObject\\b","\\bQApplication\\b","\\bQMainWindow\\b"
        };
        for (const auto& t : types)
            m_rules.append({QRegularExpression(t), typeFmt});

        // Preprocessor
        m_rules.append({QRegularExpression("^\\s*#\\w+"), macroFmt});

        // Function definitions
        m_rules.append({QRegularExpression("\\b(\\w+)(?=\\s*\\()"), callFmt});

        // Numbers (hex, octal, float, int)
        m_rules.append({QRegularExpression("\\b0x[0-9A-Fa-f]+[uUlL]*\\b|\\b0[0-7]+[uUlL]*\\b|\\b\\d+\\.?\\d*([eE][+-]?\\d+)?[fFdDuUlL]*\\b"), numFmt});

        // Strings
        m_rules.append({QRegularExpression("L?\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"|'[^'\\\\]*(\\\\.[^'\\\\]*)*'"), strFmt});

        // Line comments
        m_rules.append({QRegularExpression("//[^\n]*"), cmtFmt});
        // Block comment start marker (multiline handled separately)
        m_multilineCommentStart = QRegularExpression("/\\*");
        m_multilineCommentEnd   = QRegularExpression("\\*/");

    // ─────────────────────────────────────────────────────────────────────────
    // JAVASCRIPT / TYPESCRIPT
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "javascript" || language == "typescript") {
        const QStringList keywords = {
            "\\bbreak\\b","\\bcase\\b","\\bcatch\\b","\\bclass\\b","\\bconst\\b",
            "\\bcontinue\\b","\\bdebugger\\b","\\bdefault\\b","\\bdelete\\b",
            "\\bdo\\b","\\belse\\b","\\bexport\\b","\\bextends\\b","\\bfinally\\b",
            "\\bfor\\b","\\bfunction\\b","\\bif\\b","\\bimport\\b","\\bin\\b",
            "\\binstanceof\\b","\\blet\\b","\\bnew\\b","\\breturn\\b","\\bstatic\\b",
            "\\bsuper\\b","\\bswitch\\b","\\bthis\\b","\\bthrow\\b","\\btry\\b",
            "\\btypeof\\b","\\bvar\\b","\\bvoid\\b","\\bwhile\\b","\\bwith\\b",
            "\\byield\\b","\\basync\\b","\\bawait\\b","\\bof\\b","\\bfrom\\b",
            "\\btrue\\b","\\bfalse\\b","\\bnull\\b","\\bundefined\\b","\\bNaN\\b",
            // TypeScript extras
            "\\binterface\\b","\\btype\\b","\\benum\\b","\\babstract\\b",
            "\\bimplements\\b","\\breadonly\\b","\\bdeclare\\b","\\bnamespace\\b",
            "\\bmodule\\b","\\bprivate\\b","\\bpublic\\b","\\bprotected\\b",
            "\\bnever\\b","\\bany\\b","\\bunknown\\b","\\bkeyof\\b","\\binfer\\b"
        };
        for (const auto& kw : keywords)
            m_rules.append({QRegularExpression(kw), kwFmt});

        // Built-ins
        const QStringList builtins = {
            "\\bconsole\\b","\\bdocument\\b","\\bwindow\\b","\\bprocess\\b",
            "\\bArray\\b","\\bObject\\b","\\bString\\b","\\bNumber\\b","\\bBoolean\\b",
            "\\bPromise\\b","\\bMap\\b","\\bSet\\b","\\bWeakMap\\b","\\bWeakSet\\b",
            "\\bError\\b","\\bRegExp\\b","\\bDate\\b","\\bMath\\b","\\bJSON\\b",
            "\\bsetTimeout\\b","\\bsetInterval\\b","\\bfetch\\b","\\brequire\\b","\\bmodule\\b"
        };
        for (const auto& b : builtins)
            m_rules.append({QRegularExpression(b), typeFmt});

        // function / arrow function names
        m_rules.append({QRegularExpression("(?<=\\bfunction\\s)\\w+"), defFmt});
        m_rules.append({QRegularExpression("\\b(\\w+)(?=\\s*[=:]\\s*(?:function|\\()\\s*=>?)"), defFmt});
        m_rules.append({QRegularExpression("\\b(\\w+)(?=\\s*\\()"), callFmt});

        // Decorators (@decorator)
        m_rules.append({QRegularExpression("@[\\w.]+"), decorFmt});

        // Template literals handled as string
        m_rules.append({QRegularExpression("`[^`]*`"), strFmt});

        // Numbers
        m_rules.append({QRegularExpression("\\b\\d+\\.?\\d*([eE][+-]?\\d+)?n?\\b|0x[0-9A-Fa-f]+n?\\b|0o[0-7]+n?\\b|0b[01]+n?\\b"), numFmt});

        // Strings
        m_rules.append({QRegularExpression("\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"|'[^'\\\\]*(\\\\.[^'\\\\]*)*'"), strFmt});

        // Comments
        m_rules.append({QRegularExpression("//[^\n]*"), cmtFmt});
        m_multilineCommentStart = QRegularExpression("/\\*");
        m_multilineCommentEnd   = QRegularExpression("\\*/");

    // ─────────────────────────────────────────────────────────────────────────
    // HTML
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "html") {
        // HTML tags
        m_rules.append({QRegularExpression("</?[a-zA-Z][a-zA-Z0-9]*"), tagFmt});
        m_rules.append({QRegularExpression(">"), tagFmt});

        // Attribute names
        m_rules.append({QRegularExpression("\\s[a-zA-Z_:][a-zA-Z0-9_:\\-\\.]*(?=\\s*=)"), attrNameFmt});

        // Attribute values
        m_rules.append({QRegularExpression("=['\"][^'\"]*['\"]"), strFmt});
        m_rules.append({QRegularExpression("\"[^\"]*\"|'[^']*'"), strFmt});

        // Comments
        m_multilineCommentStart = QRegularExpression("<!--");
        m_multilineCommentEnd   = QRegularExpression("-->");

    // ─────────────────────────────────────────────────────────────────────────
    // CSS
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "css") {
        // Selectors
        m_rules.append({QRegularExpression("[.#]?[a-zA-Z_][a-zA-Z0-9_\\-]*(?=\\s*\\{)"), selectorFmt});
        m_rules.append({QRegularExpression(":[a-z\\-]+(\\([^)]*\\))?"), decorFmt});  // pseudo

        // Properties
        m_rules.append({QRegularExpression("[a-zA-Z\\-]+(?=\\s*:)"), propFmt});

        // Values / units
        m_rules.append({QRegularExpression("\\b\\d+\\.?\\d*(?:px|em|rem|vh|vw|%|pt|deg|s|ms)?\\b"), numFmt});
        m_rules.append({QRegularExpression("#[0-9A-Fa-f]{3,8}\\b"), numFmt});

        // Strings
        m_rules.append({QRegularExpression("\"[^\"]*\"|'[^']*'"), strFmt});

        // !important
        m_rules.append({QRegularExpression("!important"), kwFmt});

        // Comments
        m_multilineCommentStart = QRegularExpression("/\\*");
        m_multilineCommentEnd   = QRegularExpression("\\*/");

    // ─────────────────────────────────────────────────────────────────────────
    // SHELL / BASH
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "shell") {
        const QStringList keywords = {
            "\\bif\\b","\\bthen\\b","\\belse\\b","\\belif\\b","\\bfi\\b",
            "\\bfor\\b","\\bdo\\b","\\bdone\\b","\\bwhile\\b","\\buntil\\b",
            "\\bcase\\b","\\besac\\b","\\bfunction\\b","\\breturn\\b",
            "\\bexport\\b","\\blocal\\b","\\breadonly\\b","\\bunset\\b",
            "\\bsource\\b","\\bin\\b","\\bselect\\b","\\bbreak\\b","\\bcontinue\\b"
        };
        for (const auto& kw : keywords)
            m_rules.append({QRegularExpression(kw), kwFmt});

        const QStringList cmds = {
            "\\becho\\b","\\bcat\\b","\\bls\\b","\\bmv\\b","\\bcp\\b","\\brm\\b",
            "\\bgrep\\b","\\bsed\\b","\\bawk\\b","\\bsort\\b","\\buniq\\b",
            "\\bcut\\b","\\bhead\\b","\\btail\\b","\\bwc\\b","\\bfind\\b",
            "\\bchmod\\b","\\bchown\\b","\\bcd\\b","\\bpwd\\b","\\bmkdir\\b",
            "\\btouch\\b","\\btest\\b","\\bread\\b","\\bprintf\\b","\\bexec\\b",
            "\\bpython3\\b","\\bpython\\b","\\bbash\\b","\\bsh\\b","\\bsudo\\b"
        };
        for (const auto& cmd : cmds)
            m_rules.append({QRegularExpression(cmd), typeFmt});

        // Variables $VAR / ${VAR}
        m_rules.append({QRegularExpression("\\$\\{?[A-Za-z_][A-Za-z0-9_]*\\}?"), selfFmt});
        m_rules.append({QRegularExpression("\\$[0-9@*#?$!\\-]"), selfFmt});

        // Strings
        m_rules.append({QRegularExpression("\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"|'[^'\\\\]*(\\\\.[^'\\\\]*)*'"), strFmt});

        // Numbers
        m_rules.append({QRegularExpression("\\b\\d+\\b"), numFmt});

        // Comments
        m_rules.append({QRegularExpression("#[^\n]*"), cmtFmt});

    // ─────────────────────────────────────────────────────────────────────────
    // JSON
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "json") {
        // Keys (property names)
        m_rules.append({QRegularExpression("\"[^\"]+\"(?=\\s*:)"), attrNameFmt});
        // Values - strings
        m_rules.append({QRegularExpression("(?<=:\\s*)\"[^\"]*\""), strFmt});
        // Numbers
        m_rules.append({QRegularExpression("(?<=:\\s*)\\-?\\d+\\.?\\d*([eE][+-]?\\d+)?\\b"), numFmt});
        // Booleans / null
        m_rules.append({QRegularExpression("\\btrue\\b|\\bfalse\\b|\\bnull\\b"), kwFmt});
        // Any remaining strings
        m_rules.append({QRegularExpression("\"[^\"]*\""), strFmt});

    // ─────────────────────────────────────────────────────────────────────────
    // YAML
    // ─────────────────────────────────────────────────────────────────────────
    } else if (language == "yaml") {
        // Keys
        m_rules.append({QRegularExpression("^\\s*[\\w\\-]+(?=\\s*:)"), attrNameFmt});
        // Special values
        m_rules.append({QRegularExpression("\\b(true|false|null|yes|no|on|off)\\b"), kwFmt});
        // Anchors / aliases
        m_rules.append({QRegularExpression("[&*][\\w]+"), decorFmt});
        // Tags
        m_rules.append({QRegularExpression("![\\w!]+"), decorFmt});
        // Numbers
        m_rules.append({QRegularExpression("\\b\\d+\\.?\\d*\\b"), numFmt});
        // Strings
        m_rules.append({QRegularExpression("\"[^\"]*\"|'[^']*'"), strFmt});
        // Comments
        m_rules.append({QRegularExpression("#[^\n]*"), cmtFmt});

    // ─────────────────────────────────────────────────────────────────────────
    // FALLBACK
    // ─────────────────────────────────────────────────────────────────────────
    } else {
        m_rules.append({QRegularExpression("//[^\n]*|#[^\n]*"), cmtFmt});
        m_rules.append({QRegularExpression("\"[^\"]*\"|'[^']*'"), strFmt});
        m_rules.append({QRegularExpression("\\b\\d+\\.?\\d*\\b"), numFmt});
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// highlightBlock
// ─────────────────────────────────────────────────────────────────────────────
void SyntaxHighlighter::highlightBlock(const QString& text)
{
    // ── Multi-line block comment handling (C, C++, JS, CSS, HTML) ────────────
    if (m_multilineCommentStart.pattern().size() > 0) {
        QTextCharFormat cmtFmt;
        cmtFmt.setForeground(QColor("#6a9955"));
        cmtFmt.setFontItalic(true);

        int prev = previousBlockState();
        if (prev == 10) {
            // We are inside a block comment
            QRegularExpressionMatch endMatch = m_multilineCommentEnd.match(text);
            if (endMatch.hasMatch()) {
                int endPos = endMatch.capturedStart() + endMatch.capturedLength();
                setFormat(0, endPos, cmtFmt);
                setCurrentBlockState(0);
                // Continue scanning from endPos
            } else {
                setFormat(0, text.length(), cmtFmt);
                setCurrentBlockState(10);
                return;
            }
        }

        // Look for opening of block comment
        QRegularExpressionMatchIterator startIter = m_multilineCommentStart.globalMatch(text);
        while (startIter.hasNext()) {
            auto startMatch = startIter.next();
            int sPos = startMatch.capturedStart();
            QRegularExpressionMatch endMatch = m_multilineCommentEnd.match(text, sPos + startMatch.capturedLength());
            if (endMatch.hasMatch()) {
                int ePos = endMatch.capturedStart() + endMatch.capturedLength();
                setFormat(sPos, ePos - sPos, cmtFmt);
            } else {
                setFormat(sPos, text.length() - sPos, cmtFmt);
                setCurrentBlockState(10);
                return;
            }
        }
    }

    // ── Python triple-quote multi-line strings ────────────────────────────────
    if (m_tripleQuoteSingle.pattern().size() > 0) {
        int prev = previousBlockState();
        if (prev == 1 || prev == 2) {
            QRegularExpression end = (prev == 1) ? m_tripleQuoteSingle : m_tripleQuoteDouble;
            QRegularExpressionMatch m = end.match(text);
            if (m.hasMatch()) {
                int endPos = m.capturedStart() + 3;
                setFormat(0, endPos, m_stringFormat);
                setCurrentBlockState(0);
            } else {
                setFormat(0, text.length(), m_stringFormat);
                setCurrentBlockState(prev);
                return;
            }
        }

        for (int i = 0; i < text.length() - 2; ) {
            bool found = false;
            for (int q = 1; q <= 2; q++) {
                QRegularExpression open = (q == 1) ? m_tripleQuoteSingle : m_tripleQuoteDouble;
                QRegularExpressionMatch mo = open.match(text, i);
                if (mo.hasMatch() && mo.capturedStart() == i) {
                    QRegularExpressionMatch mc = open.match(text, i + 3);
                    if (mc.hasMatch()) {
                        int len = mc.capturedStart() + 3 - i;
                        setFormat(i, len, m_stringFormat);
                        i += len;
                    } else {
                        setFormat(i, text.length() - i, m_stringFormat);
                        setCurrentBlockState(q);
                        return;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) i++;
        }
    }

    // ── Single-line rules ─────────────────────────────────────────────────────
    for (const Rule& rule : m_rules) {
        QRegularExpressionMatchIterator it = rule.pattern.globalMatch(text);
        while (it.hasNext()) {
            QRegularExpressionMatch m = it.next();
            setFormat(m.capturedStart(), m.capturedLength(), rule.format);
        }
    }

    if (previousBlockState() <= 0)
        setCurrentBlockState(0);
}
