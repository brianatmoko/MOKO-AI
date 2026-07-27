#include "chat_widget.h"
#include "stress_test_dialog.h"
#include "graphify_dialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QNetworkRequest>
#include <QScrollBar>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDateTime>
#include <QUuid>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QKeyEvent>
#include <QPoint>
#include <QProcess>
#include <algorithm>

// ── Slash Commands Table ──────────────────────────────────────────────────────
static const QVector<QPair<QString,QString>> SLASH_COMMANDS = {
    { "/coding",          "Bantu coding / edit file yang aktif di editor" },
    { "/belajar",         "Belajar topik secara otonom" },
    { "/belajar-darkweb", "Belajar dari sumber darkweb" },
    { "/learn_logic",     "Belajar logika / matematika" },
    { "/inject_folder",   "Injeksi folder ke memori MOKO" },
    { "/play",            "Buat & jalankan kode sementara (playground)" },
    { "/repair",          "Self-repair GUI atau kode MOKO" },
    { "/stop",            "Hentikan proses belajar / crawl" },
    { "/dataset",         "Unduh dataset pengetahuan" },
    { "/forager",         "Kontrol autonomous data forager" },
    { "/path",            "Tampilkan roadmap pembelajaran" },
    { "/clearmem",        "Bersihkan session buffer" },
    { "/stress",          "Buka MOKO Kognitif Stress-Test Console (100K LOC)" },
    { "/graphify",        "Monitor jaringan, memori, & RAG secara visual" },
    { "/help",            "Tampilkan semua perintah slash" },
};

// ── Helper: sessions dir ──────────────────────────────────────────────────────
static QString sessionsDirPath() {
    return QDir::homePath() + "/.moko_sessions";
}

// ── Helper: sender color ──────────────────────────────────────────────────────
static QString senderColor(const QString& sender) {
    if (sender.startsWith("Mandor"))        return "#ffd700";
    if (sender.startsWith("Pekerja"))       return "#ff6b35";
    if (sender.startsWith("Guard"))         return "#00ff88";
    if (sender.startsWith("Moko-Local"))    return "#cc44ff";
    if (sender.startsWith("Moko-Rotation")) return "#ff6b35";
    if (sender.contains("MANDOR"))          return "#ffd700";
    if (sender.contains("VALIDATOR"))       return "#ff8c00";
    if (sender.contains("EKSEKUTOR"))       return "#ff6b35";
    if (sender.contains("Phase 4"))         return "#00ff88";
    if (sender.contains("Belajar"))         return "#ffd700";
    return "#cc44ff";
}

// ── Helper: format message with basic Markdown and Think/Code blocks ──────────
static QString formatMessageToHtml(const QString& rawText) {
    QString text = rawText;
    
    // 1. Ekstrak dan format blok <think>...</think>
    int thinkStart = text.indexOf("<think>");
    while (thinkStart != -1) {
        int thinkEnd = text.indexOf("</think>", thinkStart + 7);
        if (thinkEnd != -1) {
            QString thinkingContent = text.mid(thinkStart + 7, thinkEnd - (thinkStart + 7)).trimmed();
            thinkingContent.replace("\n", "<br>");
            
            QString thinkHtml = QString(
                "<div style='background: rgba(255,107,53,0.02); "
                "border-left: 3px dashed rgba(255,107,53,0.22); "
                "padding: 8px 12px; margin: 8px 0; "
                "color: rgba(160,180,210,0.6); font-style: italic; "
                "font-size: 11px; font-family: sans-serif;'>"
                "<b style='color: rgba(255,107,53,0.65); font-style: normal; "
                "font-family: \"Fira Code\", monospace;'>◈ PROSES BERPIKIR MOKO</b><br>%1"
                "</div>"
            ).arg(thinkingContent);
            
            text.replace(thinkStart, (thinkEnd + 8) - thinkStart, thinkHtml);
            thinkStart = text.indexOf("<think>", thinkStart + thinkHtml.length());
        } else {
            QString thinkingContent = text.mid(thinkStart + 7).trimmed();
            thinkingContent.replace("\n", "<br>");
            QString thinkHtml = QString(
                "<div style='background: rgba(255,107,53,0.02); "
                "border-left: 3px dashed rgba(255,107,53,0.22); "
                "padding: 8px 12px; margin: 8px 0; "
                "color: rgba(160,180,210,0.6); font-style: italic; "
                "font-size: 11px; font-family: sans-serif;'> "
                "<b style='color: rgba(255,107,53,0.65); font-style: normal; "
                "font-family: \"Fira Code\", monospace;'>◈ PROSES BERPIKIR MOKO (sedang berpikir...)</b><br>%1"
                "</div>"
            ).arg(thinkingContent);
            text.replace(thinkStart, text.length() - thinkStart, thinkHtml);
            break;
        }
    }

    // 2. Ekstrak dan format blok kode (```` ```lang ... ``` ````)
    int codeStart = text.indexOf("```");
    while (codeStart != -1) {
        int nextNewline = text.indexOf("\n", codeStart + 3);
        int codeEnd = text.indexOf("```", codeStart + 3);
        if (codeEnd != -1) {
            int contentStart = (nextNewline != -1 && nextNewline < codeEnd) ? nextNewline + 1 : codeStart + 3;
            QString codeLang = "";
            if (nextNewline != -1 && nextNewline < codeEnd) {
                codeLang = text.mid(codeStart + 3, nextNewline - (codeStart + 3)).trimmed();
            }
            QString codeContent = text.mid(contentStart, codeEnd - contentStart).trimmed();
            codeContent = codeContent.toHtmlEscaped();
            codeContent.replace("\n", "<br>");
            
            QString langHeader = codeLang.isEmpty() ? "TERMINAL" : codeLang.toUpper() + " TERMINAL";
            QString terminalHtml = QString(
                "<div style='margin: 10px 0; border-radius: 6px; overflow: hidden; "
                "border: 1px solid rgba(255,107,53,0.5); background-color: #050505;'>"
                "  <div style='background-color: #111111; color: #ff6b35; "
                "  font-family: \"Fira Code\", monospace; font-size: 9px; "
                "  padding: 4px 10px; font-weight: bold; border-bottom: 1px solid #1c1c1c;'>"
                "    ● ● ● &nbsp;&nbsp;&nbsp; %1"
                "  </div>"
                "  <div style='padding: 10px; font-family: \"Fira Code\", monospace; "
                "  font-size: 11px; color: #00ff88; line-height: 1.4; overflow-x: auto; white-space: pre-wrap;'>"
                "    %2"
                "  </div>"
                "</div>"
            ).arg(langHeader, codeContent);
            
            text.replace(codeStart, (codeEnd + 3) - codeStart, terminalHtml);
            codeStart = text.indexOf("```", codeStart + terminalHtml.length());
        } else {
            int contentStart = (nextNewline != -1) ? nextNewline + 1 : codeStart + 3;
            QString codeLang = "";
            if (nextNewline != -1) {
                codeLang = text.mid(codeStart + 3, nextNewline - (codeStart + 3)).trimmed();
            }
            QString codeContent = text.mid(contentStart).trimmed();
            codeContent = codeContent.toHtmlEscaped();
            codeContent.replace("\n", "<br>");
            
            QString langHeader = codeLang.isEmpty() ? "TERMINAL" : codeLang.toUpper() + " TERMINAL (mengetik...)";
            QString terminalHtml = QString(
                "<div style='margin: 10px 0; border-radius: 6px; overflow: hidden; "
                "border: 1px solid rgba(255,107,53,0.5); background-color: #050505;'>"
                "  <div style='background-color: #111111; color: #ff6b35; "
                "  font-family: \"Fira Code\", monospace; font-size: 9px; "
                "  padding: 4px 10px; font-weight: bold; border-bottom: 1px solid #1c1c1c;'>"
                "    ● ● ● &nbsp;&nbsp;&nbsp; %1"
                "  </div>"
                "  <div style='padding: 10px; font-family: \"Fira Code\", monospace; "
                "  font-size: 11px; color: #00ff88; line-height: 1.4; overflow-x: auto; white-space: pre-wrap;'>"
                "    %2"
                "  </div>"
                "</div>"
            ).arg(langHeader, codeContent);
            text.replace(codeStart, text.length() - codeStart, terminalHtml);
            break;
        }
    }

    // 3. Konversi format markdown bold (**bold**) -> <b>bold</b>
    int boldStart = text.indexOf("**");
    bool isBoldOpen = false;
    while (boldStart != -1) {
        if (!isBoldOpen) {
            text.replace(boldStart, 2, "<b>");
            isBoldOpen = true;
        } else {
            text.replace(boldStart, 2, "</b>");
            isBoldOpen = false;
        }
        boldStart = text.indexOf("**", boldStart + 3);
    }
    if (isBoldOpen) {
        text += "</b>";
    }

    // 4. Konversi format markdown italic (*italic*) -> <i>italic</i>
    int italicStart = text.indexOf("*");
    bool isItalicOpen = false;
    while (italicStart != -1) {
        if (italicStart < text.length() - 1 && text.at(italicStart + 1) == '*') {
            italicStart = text.indexOf("*", italicStart + 2);
            continue;
        }
        bool isListIndicator = false;
        if (italicStart == 0) isListIndicator = true;
        else {
            QChar prevChar = text.at(italicStart - 1);
            if (prevChar == '\n' || prevChar == ' ' || prevChar == '\t' || prevChar == '>') {
                if (italicStart < text.length() - 1 && text.at(italicStart + 1) == ' ') {
                    isListIndicator = true;
                }
            }
        }

        if (isListIndicator) {
            text.replace(italicStart, 1, "&bull;");
            italicStart = text.indexOf("*", italicStart + 6);
            continue;
        }

        if (!isItalicOpen) {
            text.replace(italicStart, 1, "<i>");
            isItalicOpen = true;
            italicStart = text.indexOf("*", italicStart + 3);
        } else {
            text.replace(italicStart, 1, "</i>");
            isItalicOpen = false;
            italicStart = text.indexOf("*", italicStart + 4);
        }
    }
    if (isItalicOpen) {
        text += "</i>";
    }

    // 5. Konversi format markdown header (###, ##, #) secara rapi
    QStringList lines = text.split("\n");
    for (int i = 0; i < lines.size(); ++i) {
        QString line = lines[i].trimmed();
        if (line.startsWith("### ")) {
            lines[i] = QString("<h4 style='color:#ff6b35; margin:10px 0 4px 0; font-weight:bold; font-family:sans-serif;'>%1</h4>")
                       .arg(line.mid(4));
        } else if (line.startsWith("## ")) {
            lines[i] = QString("<h3 style='color:#ff6b35; margin:14px 0 6px 0; font-weight:bold; font-family:sans-serif;'>%1</h3>")
                       .arg(line.mid(3));
        } else if (line.startsWith("# ")) {
            lines[i] = QString("<h2 style='color:#ff6b35; margin:18px 0 8px 0; font-weight:bold; font-family:sans-serif;'>%1</h2>")
                       .arg(line.mid(2));
        }
    }
    text = lines.join("\n");

    // 6. Menyederhanakan rumus LaTeX matematika agar ramah manusia biasa
    text.replace("\\cdot", " * ");
    text.replace("\\times", " * ");
    text.replace("\\int", " ∫ ");
    text.replace("\\,", " ");
    text.replace("\\ln", "ln");
    
    // Konversi fraksi LaTeX: \frac{numerator}{denominator} -> (numerator) / (denominator)
    int fracIdx = text.indexOf("\\frac{");
    while (fracIdx != -1) {
        int firstBraceOpen = fracIdx + 5;
        int firstBraceClose = -1;
        int braceDepth = 0;
        for (int i = firstBraceOpen; i < text.length(); ++i) {
            if (text.at(i) == '{') braceDepth++;
            else if (text.at(i) == '}') {
                if (braceDepth == 0) { firstBraceClose = i; break; }
                else braceDepth--;
            }
        }
        
        if (firstBraceClose != -1 && firstBraceClose < text.length() - 2 && text.at(firstBraceClose + 1) == '{') {
            int secondBraceOpen = firstBraceClose + 1;
            int secondBraceClose = -1;
            braceDepth = 0;
            for (int i = secondBraceOpen + 1; i < text.length(); ++i) {
                if (text.at(i) == '{') braceDepth++;
                else if (text.at(i) == '}') {
                    if (braceDepth == 0) { secondBraceClose = i; break; }
                    else braceDepth--;
                }
            }
            
            if (secondBraceClose != -1) {
                QString numerator = text.mid(firstBraceOpen + 1, firstBraceClose - (firstBraceOpen + 1));
                QString denominator = text.mid(secondBraceOpen + 1, secondBraceClose - (secondBraceOpen + 1));
                
                QString simpleFrac;
                if (numerator.length() <= 2 && denominator.length() <= 2) {
                    simpleFrac = QString("%1/%2").arg(numerator, denominator);
                } else {
                    simpleFrac = QString("(%1) / (%2)").arg(numerator, denominator);
                }
                
                text.replace(fracIdx, (secondBraceClose + 1) - fracIdx, simpleFrac);
                fracIdx = text.indexOf("\\frac{", fracIdx + simpleFrac.length());
                continue;
            }
        }
        fracIdx = text.indexOf("\\frac{", fracIdx + 6);
    }

    // Hapus sisa-sisa simbol LaTeX bracket atau dollar yang mengganggu
    text.replace("$$", "");
    text.replace("$", "");

    // 7. Konversi tabel markdown (| col | col |) menjadi tabel HTML bergaya
    {
        QStringList lines = text.split("\n");
        QStringList result;
        int i = 0;
        while (i < lines.size()) {
            QString line = lines[i].trimmed();
            // Deteksi baris tabel: dimulai dan diakhiri dengan |
            if (line.startsWith("|") && line.endsWith("|")) {
                // Mulai tabel
                QString tableHtml = "<table style='border-collapse:collapse; margin:10px 0; width:100%; "
                                    "font-size:12px; font-family:sans-serif;'>";
                bool firstRow = true;
                while (i < lines.size()) {
                    QString tline = lines[i].trimmed();
                    if (!tline.startsWith("|") || !tline.endsWith("|")) break;
                    // Lewati baris separator (|---|---|)
                    if (tline.contains("---") || tline.contains("===")) { i++; continue; }
                    QStringList cols = tline.split("|");
                    // Hapus elemen kosong dari split
                    if (!cols.isEmpty() && cols.first().trimmed().isEmpty()) cols.removeFirst();
                    if (!cols.isEmpty() && cols.last().trimmed().isEmpty()) cols.removeLast();

                    if (firstRow) {
                        tableHtml += "<tr style='background:rgba(255,107,53,0.12);'>";
                        for (const auto& col : cols) {
                            tableHtml += QString("<th style='border:1px solid rgba(255,107,53,0.3); "
                                                 "padding:6px 10px; color:#ff6b35; "
                                                 "text-align:left;'>%1</th>").arg(col.trimmed());
                        }
                        tableHtml += "</tr>";
                        firstRow = false;
                    } else {
                        tableHtml += "<tr style='background:rgba(255,255,255,0.02);'>";
                        for (const auto& col : cols) {
                            tableHtml += QString("<td style='border:1px solid rgba(255,107,53,0.15); "
                                                 "padding:5px 10px; color:#e8d8ff;'>%1</td>").arg(col.trimmed());
                        }
                        tableHtml += "</tr>";
                    }
                    i++;
                }
                tableHtml += "</table>";
                result.append(tableHtml);
            } else {
                result.append(lines[i]);
                i++;
            }
        }
        text = result.join("\n");
    }

    text.replace("\n", "<br>");
    return text;
}


// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

ChatWidget::ChatWidget(QWidget* parent)
    : QWidget(parent)
    , m_networkManager(new QNetworkAccessManager(this))
{
    // ── m_sessionList hidden helper (kept for load/save session logic) ────────
    m_sessionList = new QListWidget(this);
    m_sessionList->setVisible(false);
    m_sessionList->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_sessionList, &QListWidget::itemClicked, this, &ChatWidget::onSessionSelected);
    connect(m_sessionList, &QListWidget::customContextMenuRequested,
            this, &ChatWidget::showSessionContextMenu);

    // ── Root layout: just chatContainer, full width ───────────────────────────
    m_chatContainer = new QWidget(this);
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);
    rootLayout->addWidget(m_chatContainer);

    QVBoxLayout* chatLayout = new QVBoxLayout(m_chatContainer);
    chatLayout->setContentsMargins(0, 0, 0, 0);
    chatLayout->setSpacing(0);

    // ── Header ────────────────────────────────────────────────────────────────
    QWidget* header = new QWidget(m_chatContainer);
    header->setStyleSheet(
        "background: #0d0d0d; "
        "border-bottom: 1px solid #1a1a1a;"
    );
    header->setFixedHeight(32);
    QHBoxLayout* headerRow = new QHBoxLayout(header);
    headerRow->setContentsMargins(10, 0, 6, 0);
    headerRow->setSpacing(6);

    QLabel* title = new QLabel("MOKO AI ASSISTANT", header);
    title->setStyleSheet(
        "color: #ff6b35; font-size: 10px; font-weight: 700; "
        "letter-spacing: 1.5px; font-family: 'Fira Code',monospace;"
    );
    headerRow->addWidget(title);
    headerRow->addStretch();

    // ── History dropdown button ────────────────────────────────────────────────
    m_btnHistory = new QPushButton("🕒 History", header);
    m_btnHistory->setFixedHeight(22);
    m_btnHistory->setStyleSheet(
        "QPushButton { background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.22); "
        "color: rgba(255,107,53,0.75); font-size: 9px; font-weight: 600; "
        "border-radius: 4px; padding: 0 8px; font-family: 'Fira Code',monospace; }"
        "QPushButton:hover { background: rgba(255,107,53,0.18); border-color: rgba(255,107,53,0.55); color: #ff6b35; }"
        "QPushButton:pressed { background: rgba(255,107,53,0.28); }"
    );
    connect(m_btnHistory, &QPushButton::clicked, this, &ChatWidget::showHistoryMenu);
    headerRow->addWidget(m_btnHistory);

    // ── New Session button ─────────────────────────────────────────────────────
    QPushButton* btnNew = new QPushButton("⊕ New", header);
    btnNew->setFixedHeight(22);
    btnNew->setStyleSheet(
        "QPushButton { background: rgba(255,107,53,0.10); border: 1px solid rgba(255,107,53,0.28); "
        "color: rgba(255,107,53,0.8); font-size: 9px; font-weight: 600; "
        "border-radius: 4px; padding: 0 8px; font-family: 'Fira Code',monospace; }"
        "QPushButton:hover { background: rgba(255,107,53,0.22); border-color: rgba(255,107,53,0.6); color: #ff6b35; }"
        "QPushButton:pressed { background: rgba(255,107,53,0.32); }"
    );
    connect(btnNew, &QPushButton::clicked, this, [this]() { newSession(); });
    headerRow->addWidget(btnNew);

    // ── Multi-Agent checkbox ───────────────────────────────────────────────────
    m_chkAgentMode = new QCheckBox("🤖", header);
    m_chkAgentMode->setToolTip("Multi-Agent Mode");
    m_chkAgentMode->setStyleSheet(
        "QCheckBox { color: rgba(255,107,53,0.8); font-size: 11px; padding: 0 2px; }"
        "QCheckBox::indicator { width: 12px; height: 12px; }"
        "QCheckBox::indicator:unchecked { background: rgba(0,0,0,0.4); border: 1px solid rgba(255,107,53,0.2); border-radius: 2px; }"
        "QCheckBox::indicator:checked { background: rgba(255,107,53,0.2); border: 1px solid rgba(255,107,53,0.7); border-radius: 2px; }"
    );
    headerRow->addWidget(m_chkAgentMode);
    chatLayout->addWidget(header);

    // ── X-Ray Thinking Panel ──────────────────────────────────────────────────
    m_thinkingWidget = new QWidget(m_chatContainer);
    m_thinkingWidget->setStyleSheet(
        "background: #0a0a0a; "
        "border-bottom: 1px solid #1a1a1a;"
    );
    QVBoxLayout* thinkLayout = new QVBoxLayout(m_thinkingWidget);
    thinkLayout->setContentsMargins(12, 6, 12, 6);
    thinkLayout->setSpacing(4);

    QHBoxLayout* thinkHeaderRow = new QHBoxLayout();
    QLabel* thinkIcon = new QLabel("◈ XRAY", m_thinkingWidget);
    thinkIcon->setStyleSheet(
        "color: #ff6b35; font-size: 10px; font-weight: 700; "
        "letter-spacing: 1.5px; font-family: 'Fira Code',monospace;"
    );
    m_thinkStatus = new QLabel("analyzing...", m_thinkingWidget);
    m_thinkStatus->setStyleSheet(
        "color: rgba(255,107,53,0.55); font-size: 10px; "
        "font-family: 'Fira Code',monospace;"
    );
    m_thinkToggle = new QPushButton("▼ show", m_thinkingWidget);
    m_thinkToggle->setStyleSheet(
        "QPushButton { background: transparent; border: none; "
        "color: rgba(255,107,53,0.5); font-size: 10px; "
        "font-family: 'Fira Code',monospace; padding: 0; }"
        "QPushButton:hover { color: rgba(255,107,53,0.85); }"
    );
    m_thinkToggle->setFixedHeight(16);
    connect(m_thinkToggle, &QPushButton::clicked, this, &ChatWidget::toggleThinkingContent);
    thinkHeaderRow->addWidget(thinkIcon);
    thinkHeaderRow->addSpacing(8);
    thinkHeaderRow->addWidget(m_thinkStatus);
    thinkHeaderRow->addStretch();
    thinkHeaderRow->addWidget(m_thinkToggle);
    thinkLayout->addLayout(thinkHeaderRow);

    m_thinkingBox = new QTextEdit(m_thinkingWidget);
    m_thinkingBox->setReadOnly(true);
    m_thinkingBox->setMaximumHeight(130);
    m_thinkingBox->setVisible(false);
    m_thinkingBox->setStyleSheet(
        "QTextEdit { background: transparent; border: none; "
        "color: rgba(255,200,180,0.75); "
        "font-family: 'Fira Code',monospace; font-size: 11px; }"
    );
    thinkLayout->addWidget(m_thinkingBox);
    m_thinkingWidget->setVisible(false);
    chatLayout->addWidget(m_thinkingWidget);

    // ── Chat History area ─────────────────────────────────────────────────────
    m_chatHistory = new QTextEdit(m_chatContainer);
    m_chatHistory->setReadOnly(true);
    m_chatHistory->setStyleSheet(
        "QTextEdit { background-color: #0a0a0a; border: none; "
        "color: #d0d0d0; font-family: 'Inter','Segoe UI',sans-serif; "
        "font-size: 12px; padding: 10px 14px; }"
    );
    m_chatHistory->setPlaceholderText("Halo! Tanya Moko AI... ketik / untuk slash commands");
    chatLayout->addWidget(m_chatHistory, 1);

    // ── Input Area ────────────────────────────────────────────────────────────
    QWidget* inputArea = new QWidget(m_chatContainer);
    inputArea->setStyleSheet("background-color: #0d0d0d;");
    QVBoxLayout* inputAreaLayout = new QVBoxLayout(inputArea);
    inputAreaLayout->setContentsMargins(10, 4, 10, 8);
    inputAreaLayout->setSpacing(4);

    QLabel* hintBar = new QLabel(
        "<b style='color:#ff6b35'>/coding</b> edit file &nbsp;|&nbsp; "
        "<b style='color:#00ff88'>/belajar</b> topik &nbsp;|&nbsp; "
        "<b style='color:#ff6b35'>/</b> semua commands",
        inputArea
    );
    hintBar->setTextFormat(Qt::RichText);
    hintBar->setStyleSheet("color: rgba(140,160,200,0.35); font-size: 10px;");
    inputAreaLayout->addWidget(hintBar);

    QHBoxLayout* inputRow = new QHBoxLayout();
    inputRow->setSpacing(8);

    m_inputField = new QLineEdit(inputArea);
    m_inputField->setPlaceholderText("Ketik pertanyaan atau /command untuk Moko AI...");
    m_inputField->setStyleSheet(
        "QLineEdit { background-color: #0a0a0a; border: 1px solid #2a2a2a; "
        "border-radius: 6px; padding: 8px 12px; color: #ffffff; "
        "font-family: 'Inter',sans-serif; font-size: 12px; } "
        "QLineEdit:focus { border: 1px solid #ff6b35; background-color: #0a0a0a; }"
    );
    connect(m_inputField, &QLineEdit::returnPressed, this, &ChatWidget::sendChat);
    connect(m_inputField, &QLineEdit::textChanged, this, &ChatWidget::onInputTextChanged);
    m_inputField->installEventFilter(this);
    inputRow->addWidget(m_inputField, 1);

    m_sendButton = new QPushButton("▶", inputArea);
    m_sendButton->setFixedSize(38, 38);
    m_sendButton->setStyleSheet(
        "QPushButton { background: rgba(255,107,53,0.15); border: 1px solid rgba(255,107,53,0.4); "
        "border-radius: 6px; color: #ff6b35; font-weight: bold; font-size: 14px; } "
        "QPushButton:hover { background: rgba(255,107,53,0.3); border-color: rgba(255,107,53,0.7); }"
        "QPushButton:disabled { color: rgba(255,107,53,0.3); }"
    );
    connect(m_sendButton, &QPushButton::clicked, this, &ChatWidget::sendChat);
    inputRow->addWidget(m_sendButton);

    inputAreaLayout->addLayout(inputRow);
    setupMarathonBar();
    chatLayout->addWidget(m_marathonBar);
    chatLayout->addWidget(inputArea);

    // ── Slash Popup (floating tooltip) ────────────────────────────────────────
    m_slashPopup = new QListWidget(m_chatContainer);
    m_slashPopup->setWindowFlags(Qt::ToolTip | Qt::FramelessWindowHint);
    m_slashPopup->setFocusPolicy(Qt::NoFocus);
    m_slashPopup->setVisible(false);
    m_slashPopup->setStyleSheet(R"(
        QListWidget {
            background: rgba(13,13,13,0.99);
            border: 1px solid #ff6b35;
            border-radius: 8px;
            font-family: 'Fira Code',monospace;
            font-size: 11px;
            padding: 4px 0;
        }
        QListWidget::item {
            padding: 7px 14px;
            color: #a0a0a0;
            margin: 1px 3px;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background: rgba(255,107,53,0.18);
            color: #ff6b35;
        }
    )");
    connect(m_slashPopup, &QListWidget::itemClicked, this, &ChatWidget::onSlashItemClicked);

    // Init
    newSession();
    loadSessionList();
}

ChatWidget::~ChatWidget() {
    if (m_activeReply) m_activeReply->abort();
}

// ─────────────────────────────────────────────────────────────────────────────
// X-Ray Thinking Panel
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::startThinking(const QString& status) {
    m_thinkingActive = true;
    m_thinkingBox->clear();
    m_thinkStatus->setText(status);
    m_thinkingWidget->setVisible(true);
    m_thinkToggle->setText("▼ show");
    m_thinkingBox->setVisible(false);
    m_sendButton->setEnabled(false);
}

void ChatWidget::appendThinking(const QString& text, const QString& color) {
    if (!m_thinkingActive) return;
    QString c = color.isEmpty() ? "rgba(200,180,255,0.75)" : color;
    QString escaped = text.toHtmlEscaped().replace("\n", "<br>");
    m_thinkingBox->moveCursor(QTextCursor::End);
    m_thinkingBox->insertHtml(
        QString("<div style='margin-bottom:3px; color:%1;'>▸ %2</div>").arg(c, escaped)
    );
    m_thinkingBox->verticalScrollBar()->setValue(m_thinkingBox->verticalScrollBar()->maximum());
}

void ChatWidget::endThinking() {
    m_thinkingActive = false;
    m_thinkStatus->setText("selesai ✓");
    m_sendButton->setEnabled(true);
    // Auto-collapse after 3 seconds
    QTimer::singleShot(3000, this, [this]() {
        if (!m_thinkingActive) {
            m_thinkingWidget->setVisible(false);
            m_thinkingBox->setVisible(false);
        }
    });
}

void ChatWidget::toggleThinkingContent() {
    bool visible = m_thinkingBox->isVisible();
    m_thinkingBox->setVisible(!visible);
    m_thinkToggle->setText(visible ? "▼ show" : "▲ hide");
}

// ─────────────────────────────────────────────────────────────────────────────
// Slash Autocomplete
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::onInputTextChanged(const QString& text) {
    if (text.startsWith("/") && !text.contains(' ')) {
        QString q = text.toLower();
        QVector<QPair<QString,QString>> matches;
        for (const auto& pair : SLASH_COMMANDS) {
            if (pair.first.toLower().startsWith(q)) {
                matches.append(pair);
            }
        }
        if (!matches.isEmpty()) {
            showSlashPopup(matches);
        } else {
            hideSlashPopup();
        }
    } else {
        hideSlashPopup();
    }
}

void ChatWidget::showSlashPopup(const QVector<QPair<QString,QString>>& matches) {
    m_slashPopup->clear();
    for (const auto& pair : matches) {
        QListWidgetItem* item = new QListWidgetItem(
            QString("  %1   —   %2").arg(pair.first, pair.second)
        );
        item->setData(Qt::UserRole, pair.first);
        m_slashPopup->addItem(item);
    }
    m_slashPopup->setCurrentRow(0);

    const int itemH = 34;
    int totalH = std::min(matches.size() * itemH + 12, 220);
    m_slashPopup->setFixedWidth(m_inputField->width() + 40);
    m_slashPopup->setFixedHeight(totalH);

    // Position above input field
    QPoint inputPos = m_inputField->mapToGlobal(QPoint(0, 0));
    m_slashPopup->move(inputPos.x(), inputPos.y() - totalH - 4);
    m_slashPopup->setVisible(true);
    m_slashPopup->raise();
}

void ChatWidget::hideSlashPopup() {
    m_slashPopup->setVisible(false);
    m_slashPopup->clear();
}

void ChatWidget::onSlashItemClicked(QListWidgetItem* item) {
    if (!item) return;
    QString cmd = item->data(Qt::UserRole).toString();
    if (!cmd.isEmpty()) {
        m_inputField->setText(cmd + " ");
        hideSlashPopup();
        m_inputField->setFocus();
    }
}

// Keyboard navigation for the slash popup
bool ChatWidget::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_inputField && event->type() == QEvent::KeyPress) {
        QKeyEvent* ke = static_cast<QKeyEvent*>(event);
        if (m_slashPopup->isVisible()) {
            if (ke->key() == Qt::Key_Down) {
                int r = m_slashPopup->currentRow();
                m_slashPopup->setCurrentRow(r < m_slashPopup->count()-1 ? r+1 : 0);
                return true;
            } else if (ke->key() == Qt::Key_Up) {
                int r = m_slashPopup->currentRow();
                m_slashPopup->setCurrentRow(r > 0 ? r-1 : m_slashPopup->count()-1);
                return true;
            } else if (ke->key() == Qt::Key_Return || ke->key() == Qt::Key_Enter) {
                QListWidgetItem* cur = m_slashPopup->currentItem();
                if (cur) { onSlashItemClicked(cur); return true; }
                hideSlashPopup();
            } else if (ke->key() == Qt::Key_Escape) {
                hideSlashPopup();
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

// ─────────────────────────────────────────────────────────────────────────────
// appendMessage
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::appendMessage(const QString& sender, const QString& text) {
    // Persist to session (skip transient system messages)
    if (!m_isLoadingSession) {
        bool isTransient = (sender == "system") &&
            (text.contains("sedang berpikir") || text.contains("initialized"));
        if (!isTransient) {
            ChatMessage msg;
            msg.sender = sender;
            msg.text   = text;
            msg.ts     = QDateTime::currentSecsSinceEpoch();
            m_currentMessages.append(msg);
            saveCurrentSession();
        }
    }

    m_chatHistory->moveCursor(QTextCursor::End);

    QString formattedText;
    if (sender == "user") {
        formattedText = QString(
            "<div style='margin:6px 0; padding:8px 12px; "
            "background:rgba(0,200,255,0.07); border-left:3px solid #00e6ff; border-radius:4px;'>"
            "<span style='color:rgba(0,230,255,0.6); font-size:10px; font-family:Fira Code; "
            "font-weight:700; letter-spacing:1px;'>ANDA</span><br>"
            "<span style='color:#e0f0ff;'>%1</span></div>"
        ).arg(text.toHtmlEscaped().replace("\n", "<br>"));

    } else if (sender == "moko") {
        QString formattedBody = formatMessageToHtml(text);
        formattedText = QString(
            "<div style='margin:6px 0; padding:8px 12px; "
            "background:rgba(200,0,255,0.07); border-left:3px solid #cc00ff; border-radius:4px;'>"
            "<span style='color:rgba(200,0,255,0.7); font-size:10px; font-family:Fira Code; "
            "font-weight:700; letter-spacing:1px;'>MOKO</span><br>"
            "<span style='color:#e8d8ff;'>%1</span></div>"
        ).arg(formattedBody);

    } else if (sender == "thinking") {
        // Route to X-Ray panel
        appendThinking(text);
        return;

    } else if (sender == "system") {
        formattedText = QString(
            "<div style='margin:3px 0; padding:4px 10px;'>"
            "<span style='color:rgba(140,160,200,0.45); font-size:11px; font-family:Fira Code;'>"
            "▸ %1</span></div>"
        ).arg(text.toHtmlEscaped().replace("\n","<br>"));

    } else {
        // Multi-Agent / 4-phase colored header
        QString color = senderColor(sender);
        QString formattedBody = formatMessageToHtml(text);
        formattedText = QString(
            "<div style='margin:8px 0; padding:6px 10px; "
            "border-left:3px solid %1; background:rgba(0,0,0,0.20); border-radius:4px;'>"
            "<b style='color:%1;'>▶ %2</b><br>%3</div>"
        ).arg(color, sender.toHtmlEscaped(), formattedBody);
    }


    m_chatHistory->insertHtml(formattedText);
    m_chatHistory->insertPlainText("\n");
    m_chatHistory->verticalScrollBar()->setValue(m_chatHistory->verticalScrollBar()->maximum());
}

void ChatWidget::setCodeContext(const QString& context) {
    m_codeContext = context;
}

// ─────────────────────────────────────────────────────────────────────────────
// sendChat — dispatch to standard or agent endpoint
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::sendChat() {
    QString prompt = m_inputField->text().trimmed();
    if (prompt.isEmpty()) return;

    hideSlashPopup();

    // Handle built-in slash commands
    if (prompt.toLower() == "/help") {
        QString helpText = "**Slash Commands tersedia:**\n";
        for (const auto& pair : SLASH_COMMANDS) {
            helpText += QString("  %1  —  %2\n").arg(pair.first, pair.second);
        }
        m_inputField->clear();
        appendMessage("user", prompt);
        appendMessage("system", helpText);
        return;
    }
    if (prompt.toLower() == "/clearmem") {
        m_inputField->clear();
        appendMessage("user", prompt);
        newSession();
        appendMessage("system", "Session buffer dibersihkan.");
        return;
    }

    // ── /stress — MOKO Kognitif Stress-Test Console ────────────────────────
    if (prompt.toLower() == "/stress") {
        m_inputField->clear();
        appendMessage("user", prompt);
        appendMessage("system",
            "\u26a1 Membuka MOKO Kognitif Stress-Test Console...\n"
            "Panel pengujian kecerdasan model 1B di tingkat industri (100K+ LOC)."
        );
        MokoStressTestDialog* dlg = new MokoStressTestDialog(this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->show();
        appendMessage("system", "\u2705 Stress-Test Console native dibuka.");
        return;
    }

    // ── /graphify — MOKO Graphify Console ──────────────────────────────────
    if (prompt.toLower() == "/graphify") {
        m_inputField->clear();
        appendMessage("user", prompt);
        appendMessage("system",
            "⚡ Membuka MOKO Graphify Console...\n"
            "Visualizer kognitif untuk memonitor struktur memori RAG, RSA, dan OMNI secara real-time."
        );
        MokoGraphifyDialog* dlg = new MokoGraphifyDialog(this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->show();
        appendMessage("system", "\u2705 Graphify Console native dibuka.");
        return;
    }

    // ── /coding — Kirim prompt dengan konteks coding ke LLM ────────────────
    if (prompt.startsWith("/coding ", Qt::CaseInsensitive)) {
        QString codingPrompt = prompt.mid(8).trimmed(); // strip "/coding "
        m_inputField->clear();
        appendMessage("user", prompt);
        // Inject coding system context ke prompt
        m_inputField->setText("[MODE KODING] " + codingPrompt);
        sendChat();
        return;
    }
    if (prompt.toLower() == "/coding") {
        m_inputField->clear();
        appendMessage("user", prompt);
        appendMessage("system", "Gunakan: /coding [deskripsi] — contoh: /coding buatkan kalkulator python");
        return;
    }

    if (m_activeReply && m_isStreaming) {
        m_activeReply->abort();
    }

    appendMessage("user", prompt);
    m_inputField->clear();
    m_sendButton->setEnabled(false);
    m_sendButton->setText("…");

    QString fullPrompt = prompt;
    if (!m_codeContext.isEmpty()) {
        fullPrompt += "\n\n[MOKO_EDITOR_CONTEXT]\n" + m_codeContext;
    }

    // Show X-Ray thinking panel
    startThinking("Memproses permintaan...");
    appendThinking("→ Parsing prompt: " + prompt.left(60) + (prompt.length() > 60 ? "..." : ""));
    appendThinking("→ Menyiapkan konteks multi-turn...");

    QJsonArray messages;
    QJsonObject sysMsg;
    sysMsg["role"] = "system";

    bool isCodingMode = fullPrompt.startsWith("[MODE KODING]");
    if (isCodingMode) {
        // Strip prefix dari fullPrompt agar tidak membingungkan LLM
        fullPrompt = fullPrompt.mid(13).trimmed();
        sysMsg["content"] = "Kamu adalah Moko, AI coding assistant expert. "
                            "Tugasmu adalah menulis kode yang bersih, lengkap, dan langsung bisa dijalankan. "
                            "Selalu berikan kode yang UTUH dan siap pakai, bukan hanya potongan snippet. "
                            "Gunakan bahasa pemrograman yang diminta user. Jika tidak disebutkan, gunakan Python. "
                            "Sertakan penjelasan singkat tentang cara kerja kode setelah kode tersebut. "
                            "Gunakan blok kode (```) untuk semua kode program. "
                            "Hindari penggunaan simbol bintang (*) untuk bold/italic kecuali daftar poin. "
                            "Jangan gunakan simbol LaTeX abstrak; gunakan '*' untuk kali dan '/' untuk bagi.";
        appendThinking("→ Mode KODING aktif — mengoptimalkan sistem untuk code generation...", "#ff6b35");
    } else {
        sysMsg["content"] = "Kamu adalah Moko, asisten AI yang cerdas untuk pemrograman, matematika, keamanan, dan pengetahuan umum. Jawab pertanyaan dengan tepat, ramah, dan profesional. "
                            "Jika menyajikan rumus matematika, terangkan arti fisik atau logika intuitif dari rumus tersebut secara sederhana agar mudah dipahami manusia biasa. "
                            "Gunakan simbol matematika dasar yang mudah dipahami manusia seperti '*' untuk perkalian dan '/' untuk pembagian (bukan hanya simbol LaTeX/cdot/frac abstrak) agar orang awam langsung mengerti cara menghitungnya. "
                            "Hindari penggunaan simbol bintang (*) atau double bintang (**) secara berlebihan untuk membungkus kata-kata biasa (bold/italic) kecuali untuk daftar poin yang rapi.";
    }

    messages.append(sysMsg);

    // ── Sertakan riwayat percakapan untuk multi-turn context ───────────────────
    // Ambil maks 4 pesan terakhir (2 turn) agar prefill cepat di GPU kecil
    int historyStart = qMax(0, m_currentMessages.size() - 4);
    for (int i = historyStart; i < m_currentMessages.size(); ++i) {
        const auto& msg = m_currentMessages[i];
        if (msg.sender == "user") {
            QJsonObject histMsg;
            histMsg["role"]    = "user";
            histMsg["content"] = msg.text;
            messages.append(histMsg);
        } else if (msg.sender == "moko" || msg.sender == "core") {
            QJsonObject histMsg;
            histMsg["role"]    = "assistant";
            histMsg["content"] = msg.text;
            messages.append(histMsg);
        }
    }

    // Pesan user terbaru
    QJsonObject userMsg;
    userMsg["role"]    = "user";
    userMsg["content"] = fullPrompt;
    messages.append(userMsg);

    if (m_chkAgentMode->isChecked()) {
        appendMessage("system", "🤖 Multi-Agent Mode aktif — Memulai kolaborasi AI...");
        appendThinking("→ Memanggil endpoint /v1/agent/chat (SSE)", "#00e6ff");

        QUrl url("http://127.0.0.1:11435/v1/agent/chat");
        QNetworkRequest req(url);
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QJsonObject payload;
        payload["messages"] = messages;
        m_activeReply = m_networkManager->post(req, QJsonDocument(payload).toJson());
        m_isStreaming = true;
        m_streamBuffer.clear();
        m_currentSender.clear();

        connect(m_activeReply, &QNetworkReply::readyRead, this, &ChatWidget::onReadyRead);
        connect(m_activeReply, &QNetworkReply::finished,  this, &ChatWidget::onFinished);
    } else {
        appendMessage("system", "Moko sedang berpikir...");
        appendThinking("→ Memanggil endpoint /v1/chat/completions (Stream)", "#00e6ff");

        QUrl url("http://127.0.0.1:11435/v1/chat/completions");
        QNetworkRequest req(url);
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QJsonObject payload;
        payload["model"]       = "moko-local-coder";
        payload["temperature"] = 0.2;
        payload["max_tokens"]  = 600;  // RTX 2050 4GB: 600 tok ~20 detik
        payload["messages"]    = messages;
        payload["stream"]      = true;
        m_activeReply = m_networkManager->post(req, QJsonDocument(payload).toJson());
        m_isStreaming = true;
        m_streamBuffer.clear();
        m_currentSender = "moko";

        if (!m_isLoadingSession) {
            ChatMessage msg;
            msg.sender = "moko";
            msg.text   = "";
            msg.ts     = QDateTime::currentSecsSinceEpoch();
            m_currentMessages.append(msg);
        }

        connect(m_activeReply, &QNetworkReply::readyRead, this, &ChatWidget::onReadyRead);
        connect(m_activeReply, &QNetworkReply::finished,  this, &ChatWidget::onFinished);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE streaming handlers
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::onReadyRead() {
    if (!m_activeReply) return;
    m_streamBuffer += m_activeReply->readAll();

    while (true) {
        int nl = m_streamBuffer.indexOf('\n');
        if (nl == -1) break;
        QByteArray line = m_streamBuffer.left(nl).trimmed();
        m_streamBuffer.remove(0, nl + 1);

        if (line.isEmpty() || line == "data: [DONE]") continue;
        if (!line.startsWith("data: ")) continue;

        QByteArray jsonPart = line.mid(6);
        QJsonDocument chunkDoc = QJsonDocument::fromJson(jsonPart);
        if (chunkDoc.isNull() || !chunkDoc.isObject()) continue;
        QJsonObject obj = chunkDoc.object();

        if (obj.contains("xray")) {
            QString step = obj["xray"].toString();
            appendThinking("⚡ " + step, "#00e6ff");
            continue;
        }

        if (obj.contains("sender")) {
            m_currentSender = obj["sender"].toString();
            QString color = obj.contains("phase_color")
                ? obj["phase_color"].toString()
                : senderColor(m_currentSender);
            m_chatHistory->moveCursor(QTextCursor::End);
            m_chatHistory->insertHtml(
                QString("<div style='margin-top:10px; padding:5px 10px; "
                        "border-left:3px solid %1; background:rgba(0,0,0,0.22); border-radius:3px;'>"
                        "<b style='color:%1;'>▶ %2</b><br></div>")
                .arg(color, m_currentSender.toHtmlEscaped())
            );
            m_chatHistory->insertPlainText("");
            appendThinking("→ Agent aktif: " + m_currentSender, color);

            if (!m_isLoadingSession) {
                ChatMessage msg;
                msg.sender = m_currentSender;
                msg.text   = "";
                msg.ts     = QDateTime::currentSecsSinceEpoch();
                m_currentMessages.append(msg);
            }
            continue;
        }

        if (obj.contains("token")) {
            QString token = obj["token"].toString();
            
            // Hapus kursor typing lama jika ada di akhir
            QTextCursor cursor = m_chatHistory->textCursor();
            cursor.movePosition(QTextCursor::End);
            cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
            if (cursor.selectedText() == "▋") {
                cursor.removeSelectedText();
            }

            // Sisipkan token baru
            m_chatHistory->moveCursor(QTextCursor::End);
            m_chatHistory->insertPlainText(token);

            // Sisipkan kursor typing baru
            m_chatHistory->moveCursor(QTextCursor::End);
            m_chatHistory->insertPlainText("▋");


            m_chatHistory->verticalScrollBar()->setValue(
                m_chatHistory->verticalScrollBar()->maximum()
            );
            if (!m_isLoadingSession && !m_currentMessages.isEmpty()) {
                m_currentMessages.last().text += token;
            }
        }

        if (obj.contains("choices")) {
            QJsonArray choices = obj["choices"].toArray();
            if (!choices.isEmpty()) {
                QJsonObject choice = choices[0].toObject();
                if (choice.contains("delta")) {
                    QJsonObject delta = choice["delta"].toObject();
                    if (delta.contains("content")) {
                        QString token = delta["content"].toString();
                        
                        // Hapus kursor typing lama jika ada di akhir
                        QTextCursor cursor = m_chatHistory->textCursor();
                        cursor.movePosition(QTextCursor::End);
                        cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
                        if (cursor.selectedText() == "▋") {
                            cursor.removeSelectedText();
                        }

                        // Sisipkan token baru
                        m_chatHistory->moveCursor(QTextCursor::End);
                        m_chatHistory->insertPlainText(token);

                        // Sisipkan kursor typing baru
                        m_chatHistory->moveCursor(QTextCursor::End);
                        m_chatHistory->insertPlainText("▋");

                        m_chatHistory->verticalScrollBar()->setValue(
                            m_chatHistory->verticalScrollBar()->maximum()
                        );
                        if (!m_isLoadingSession && !m_currentMessages.isEmpty()) {
                            m_currentMessages.last().text += token;
                        }
                    }
                }
            }
        }

        if (obj.contains("error")) {
            appendMessage("system", "⚠️ " + obj["error"].toString());
        }
    }
}

void ChatWidget::onFinished() {
    if (!m_activeReply) return;

    // Hapus kursor typing secara permanen saat respons selesai
    QTextCursor cursor = m_chatHistory->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
    if (cursor.selectedText() == "▋") {
        cursor.removeSelectedText();
    }

    if (m_activeReply->error() != QNetworkReply::NoError &&
        m_activeReply->error() != QNetworkReply::OperationCanceledError) {
        QString errMsg = QString("⚠️ Koneksi gagal: %1").arg(m_activeReply->errorString());
        appendMessage("system", errMsg);
        appendThinking("✗ " + errMsg, "#ff4444");
    } else {
        m_chatHistory->insertPlainText("\n");
        appendThinking("✓ Respons selesai diterima.", "#00ff88");
        saveCurrentSession();
        
        // ── Re-render all messages to trigger HTML Markdown & Terminal Block layout ──
        m_isLoadingSession = true;
        m_chatHistory->clear();
        appendMessage("system", "MOKO C++ AI-IDE initialized. Jantung Helper ONLINE.\nKetik /help untuk daftar perintah slash.");
        for (const auto& msg : m_currentMessages) {
            appendMessage(msg.sender, msg.text);
        }
        m_isLoadingSession = false;

        // Cek kelengkapan kode
        analyzeCompleteness();
    }


    endThinking();
    m_activeReply->deleteLater();
    m_activeReply  = nullptr;
    m_isStreaming  = false;
    m_currentSender.clear();
    m_sendButton->setEnabled(true);
    m_sendButton->setText("▶");
}

void ChatWidget::handleNetworkReply(QNetworkReply* reply) {
    if (reply == m_activeReply) return;

    if (reply->error() == QNetworkReply::NoError) {
        QByteArray data = reply->readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (!doc.isNull() && doc.isObject()) {
            QJsonObject obj = doc.object();
            QJsonArray choices = obj["choices"].toArray();
            if (!choices.isEmpty()) {
                QString content = choices[0].toObject()["message"].toObject()["content"].toString();
                appendThinking("✓ Respons diterima dari model lokal.", "#00ff88");
                appendMessage("moko", content);
            }
        }
    } else {
        QString errMsg = QString("⚠️ Gagal: %1").arg(reply->errorString());
        appendMessage("system", errMsg);
        appendThinking("✗ " + errMsg, "#ff4444");
    }

    endThinking();
    m_sendButton->setEnabled(true);
    m_sendButton->setText("▶");
    reply->deleteLater();
}

// ─────────────────────────────────────────────────────────────────────────────
// Session Management
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::newSession() {
    m_chatHistory->clear();
    m_currentMessages.clear();
    m_currentSessionId.clear();
    m_sessionList->blockSignals(true);
    m_sessionList->clearSelection();
    m_sessionList->blockSignals(false);
    appendMessage("system", "MOKO C++ AI-IDE initialized. Jantung Helper ONLINE.\nKetik /help untuk daftar perintah slash.");
}

struct SessionMeta {
    QString id;
    QString title;
    qint64  createdAt;
};

void ChatWidget::loadSessionList() {
    m_sessionList->blockSignals(true);
    m_sessionList->clear();

    QString sDir = sessionsDirPath();
    QDir().mkpath(sDir);
    QDir dir(sDir);
    QStringList files = dir.entryList({"*.json"}, QDir::Files);

    QList<SessionMeta> metas;
    for (const QString& f : files) {
        QFile file(dir.filePath(f));
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                QJsonObject obj = doc.object();
                SessionMeta m;
                m.id        = obj["id"].toString(QFileInfo(f).baseName());
                m.title     = obj["title"].toString("Sesi Baru");
                m.createdAt = obj["created_at"].toVariant().toLongLong();
                metas.append(m);
            }
            file.close();
        }
    }

    std::sort(metas.begin(), metas.end(), [](const SessionMeta& a, const SessionMeta& b) {
        return a.createdAt > b.createdAt;
    });

    for (const auto& meta : metas) {
        QListWidgetItem* item = new QListWidgetItem(meta.title, m_sessionList);
        item->setData(Qt::UserRole, meta.id);
        m_sessionList->addItem(item);
        if (meta.id == m_currentSessionId) m_sessionList->setCurrentItem(item);
    }

    m_sessionList->blockSignals(false);
}

void ChatWidget::loadSession(const QString& sessionId) {
    if (sessionId.isEmpty()) return;
    m_isLoadingSession = true;
    m_chatHistory->clear();
    m_currentMessages.clear();
    m_currentSessionId = sessionId;

    QFile file(sessionsDirPath() + "/" + sessionId + ".json");
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        if (!doc.isNull() && doc.isObject()) {
            for (const QJsonValue& v : doc.object()["messages"].toArray()) {
                QJsonObject o = v.toObject();
                ChatMessage msg;
                msg.sender = o["role"].toString("user");
                if (msg.sender == "core") msg.sender = "moko";
                msg.text = o["text"].toString();
                msg.ts   = o["ts"].toVariant().toLongLong();
                m_currentMessages.append(msg);
                appendMessage(msg.sender, msg.text);
            }
        }
        file.close();
    }

    m_isLoadingSession = false;

    m_sessionList->blockSignals(true);
    for (int i = 0; i < m_sessionList->count(); ++i) {
        QListWidgetItem* item = m_sessionList->item(i);
        if (item->data(Qt::UserRole).toString() == sessionId) {
            m_sessionList->setCurrentItem(item);
            break;
        }
    }
    m_sessionList->blockSignals(false);
}

void ChatWidget::saveCurrentSession() {
    if (m_currentMessages.isEmpty()) return;

    if (m_currentSessionId.isEmpty()) {
        m_currentSessionId = QUuid::createUuid().toString().remove('{').remove('}').left(8);
    }

    QString title = "Sesi Baru";
    for (const auto& msg : m_currentMessages) {
        if (msg.sender == "user" && !msg.text.isEmpty()) {
            QStringList words = msg.text.simplified().split(' ', Qt::SkipEmptyParts);
            if (!words.isEmpty()) {
                int take = std::min(6, words.size());
                title = QStringList(words.mid(0, take)).join(" ");
                if (words.size() > 6) title += "...";
            }
            break;
        }
    }

    QJsonObject obj;
    obj["id"]         = m_currentSessionId;
    obj["title"]      = title;
    obj["created_at"] = QDateTime::currentSecsSinceEpoch();

    QJsonArray msgArr;
    for (const auto& msg : m_currentMessages) {
        QJsonObject o;
        o["role"] = (msg.sender == "moko") ? "core" : msg.sender;
        o["text"] = msg.text;
        o["ts"]   = msg.ts;
        msgArr.append(o);
    }
    obj["messages"] = msgArr;

    QString sDir = sessionsDirPath();
    QDir().mkpath(sDir);
    QFile file(sDir + "/" + m_currentSessionId + ".json");
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        file.write(QJsonDocument(obj).toJson());
        file.close();
    }

    loadSessionList();
}

void ChatWidget::deleteSession(const QString& sessionId) {
    if (sessionId.isEmpty()) return;
    QFile::remove(sessionsDirPath() + "/" + sessionId + ".json");
    if (sessionId == m_currentSessionId) newSession();
    loadSessionList();
}

void ChatWidget::onSessionSelected(QListWidgetItem* item) {
    if (!item) return;
    loadSession(item->data(Qt::UserRole).toString());
}

void ChatWidget::showSessionContextMenu(const QPoint& pos) {
    QListWidgetItem* item = m_sessionList->itemAt(pos);
    if (!item) return;
    QString sessionId = item->data(Qt::UserRole).toString();

    QMenu menu(this);
    menu.setStyleSheet(
        "QMenu { background:#1e1e1e; color:#d4d4d4; border:1px solid #3c3c3c; }"
        "QMenu::item { padding:6px 20px; }"
        "QMenu::item:selected { background:#094771; color:#fff; }"
    );
    QAction* actLoad = menu.addAction("Buka Sesi");
    QAction* actDel  = menu.addAction("Hapus Sesi");

    QAction* sel = menu.exec(m_sessionList->mapToGlobal(pos));
    if (sel == actLoad) {
        loadSession(sessionId);
    } else if (sel == actDel) {
        if (QMessageBox::question(this, "Hapus Sesi",
                "Hapus sesi chat ini secara permanen?",
                QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
            deleteSession(sessionId);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// History Dropdown Menu  (tombol 🕒 di header)
// ─────────────────────────────────────────────────────────────────────────────
void ChatWidget::showHistoryMenu() {
    QMenu menu(this);
    menu.setStyleSheet(R"(
        QMenu {
            background: rgba(10,12,24,0.98);
            border: 1px solid rgba(0,230,255,0.22);
            border-radius: 6px;
            color: #c8d8f0;
            font-size: 11px;
            padding: 4px 0;
        }
        QMenu::item {
            padding: 7px 20px 7px 14px;
            min-width: 220px;
            border-radius: 3px;
            margin: 1px 4px;
        }
        QMenu::item:selected {
            background: rgba(0,230,255,0.12);
            color: #00e6ff;
        }
        QMenu::item:disabled {
            color: rgba(140,160,200,0.35);
        }
        QMenu::separator {
            height: 1px;
            background: rgba(0,230,255,0.10);
            margin: 4px 8px;
        }
    )");

    // ── Load sessions from disk ───────────────────────────────────────────────
    struct Meta { QString id; QString title; qint64 ts; };
    QList<Meta> metas;

    QString sDir = sessionsDirPath();
    QDir().mkpath(sDir);
    QDir dir(sDir);
    for (const QString& f : dir.entryList({"*.json"}, QDir::Files)) {
        QFile file(dir.filePath(f));
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                QJsonObject obj = doc.object();
                metas.append({
                    obj["id"].toString(QFileInfo(f).baseName()),
                    obj["title"].toString("Sesi Baru"),
                    obj["created_at"].toVariant().toLongLong()
                });
            }
            file.close();
        }
    }
    std::sort(metas.begin(), metas.end(), [](const Meta& a, const Meta& b) {
        return a.ts > b.ts;
    });

    // ── Populate menu ─────────────────────────────────────────────────────────
    if (metas.isEmpty()) {
        QAction* empty = menu.addAction("(Belum ada sesi tersimpan)");
        empty->setEnabled(false);
    } else {
        for (const auto& m : metas) {
            // Truncate very long titles
            QString label = m.title.length() > 42 ? m.title.left(39) + "…" : m.title;
            QString activeMarker = (m.id == m_currentSessionId) ? "  ●" : "";
            QAction* act = menu.addAction("🕒  " + label + activeMarker);
            // Use lambda capture to load correct session
            connect(act, &QAction::triggered, this, [this, id = m.id]() {
                loadSession(id);
            });
        }
    }

    // ── Footer actions ────────────────────────────────────────────────────────
    menu.addSeparator();
    QAction* actNew = menu.addAction("⊕  Sesi Baru");
    connect(actNew, &QAction::triggered, this, [this]() { newSession(); });

    if (!metas.isEmpty()) {
        QAction* actClear = menu.addAction("🗑  Hapus Semua History");
        connect(actClear, &QAction::triggered, this, [this, metas]() {
            if (QMessageBox::question(this, "Hapus Semua",
                    "Hapus semua sesi chat secara permanen?",
                    QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
                for (const auto& m : metas) deleteSession(m.id);
                newSession();
            }
        });
    }

    // Show below the history button
    if (m_btnHistory) {
        menu.exec(m_btnHistory->mapToGlobal(
            QPoint(0, m_btnHistory->height())));
    } else {
        menu.exec(QCursor::pos());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Marathon Auto-Continue & RAG UI Implementations
// ─────────────────────────────────────────────────────────────────────────────

void ChatWidget::setupMarathonBar() {
    m_marathonBar = new QWidget(m_chatContainer);
    m_marathonBar->setStyleSheet(
        "QWidget { background: rgba(204,68,255,0.12); "
        "border-top: 1px solid rgba(204,68,255,0.22); "
        "border-bottom: 1px solid rgba(204,68,255,0.22); }"
    );
    QHBoxLayout* layout = new QHBoxLayout(m_marathonBar);
    layout->setContentsMargins(12, 6, 12, 6);
    layout->setSpacing(10);

    m_marathonStatus = new QLabel("⚡ <b>Marathon Mode:</b> Kode belum selesai ter-generate.", m_marathonBar);
    m_marathonStatus->setStyleSheet("color: #cc44ff; font-size: 11px;");
    layout->addWidget(m_marathonStatus, 1);

    m_btnMarathonContinue = new QPushButton("Lanjutkan ⚡", m_marathonBar);
    m_btnMarathonContinue->setStyleSheet(
        "QPushButton { background: #cc44ff; border: none; color: #ffffff; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px 12px; }"
        "QPushButton:hover { background: #dd55ff; }"
    );
    connect(m_btnMarathonContinue, &QPushButton::clicked, this, &ChatWidget::continueMarathon);
    layout->addWidget(m_btnMarathonContinue);

    m_btnMarathonPaste = new QPushButton("Tempel ke Editor 📋", m_marathonBar);
    m_btnMarathonPaste->setStyleSheet(
        "QPushButton { background: #00ff88; border: none; color: #070912; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px 12px; }"
        "QPushButton:hover { background: #33ffa0; }"
    );
    connect(m_btnMarathonPaste, &QPushButton::clicked, this, &ChatWidget::pasteToEditor);
    layout->addWidget(m_btnMarathonPaste);
    m_btnMarathonPaste->hide(); // Tampil jika sudah lengkap

    m_btnMarathonStop = new QPushButton("Batal", m_marathonBar);
    m_btnMarathonStop->setStyleSheet(
        "QPushButton { background: transparent; border: 1px solid rgba(204,68,255,0.3); color: #cc44ff; font-size: 11px; border-radius: 4px; padding: 4px 12px; }"
        "QPushButton:hover { background: rgba(204,68,255,0.08); }"
    );
    connect(m_btnMarathonStop, &QPushButton::clicked, this, &ChatWidget::stopMarathon);
    layout->addWidget(m_btnMarathonStop);

    m_marathonBar->hide();
}

void ChatWidget::analyzeCompleteness() {
    if (m_currentMessages.isEmpty()) return;

    QString lastText = m_currentMessages.last().text;

    // Scan ALL code blocks and pick the longest one to get the most complete code
    int searchPos = 0;
    QString longestCodeContent;
    QString longestLanguage = "generic";

    while (true) {
        int firstBlock = lastText.indexOf("```", searchPos);
        if (firstBlock == -1) break;

        int codeStart = lastText.indexOf('\n', firstBlock);
        if (codeStart == -1) codeStart = firstBlock + 3;
        else codeStart += 1;

        int nextBlock = lastText.indexOf("```", codeStart);
        QString currentCodeContent;
        if (nextBlock != -1) {
            currentCodeContent = lastText.mid(codeStart, nextBlock - codeStart);
            searchPos = nextBlock + 3;
        } else {
            currentCodeContent = lastText.mid(codeStart);
            searchPos = lastText.length();
        }

        // Hapus penutup markdown block jika masih tertinggal di ujung
        if (currentCodeContent.endsWith("```")) {
            currentCodeContent.chop(3);
        }

        if (currentCodeContent.length() > longestCodeContent.length()) {
            longestCodeContent = currentCodeContent;
            QString tag = lastText.mid(firstBlock + 3, codeStart - (firstBlock + 3)).trimmed();
            if (!tag.isEmpty()) {
                longestLanguage = tag.split(' ').first().split('\n').first();
            }
        }
    }

    if (longestCodeContent.isEmpty()) return; // Tidak ada code block valid

    m_marathonLanguage = longestLanguage;
    m_marathonAccumulatedCode = longestCodeContent;
    
    if (m_currentMessages.size() >= 2) {
        m_marathonOriginalPrompt = m_currentMessages.at(m_currentMessages.size() - 2).text; // Pesan user terakhir
    } else {
        m_marathonOriginalPrompt = "";
    }

    // Kirim POST ke /v1/marathon/analyze
    QUrl url("http://127.0.0.1:11435/v1/marathon/analyze");
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonObject payload;
    payload["code"] = longestCodeContent;
    payload["language"] = longestLanguage;

    QNetworkReply* reply = m_networkManager->post(req, QJsonDocument(payload).toJson());
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        handleMarathonAnalysis(reply);
    });
}

void ChatWidget::handleMarathonAnalysis(QNetworkReply* reply) {
    reply->deleteLater();
    if (reply->error() != QNetworkReply::NoError) {
        appendThinking("✗ Analisis kelengkapan gagal: " + reply->errorString(), "#ff4444");
        return;
    }

    QByteArray data = reply->readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) return;

    QJsonObject obj = doc.object();
    bool complete = obj["complete"].toBool();
    QString reason = obj["reason"].toString();

    if (!complete) {
        m_marathonStatus->setText(QString("⚡ <b>Marathon Mode:</b> %1").arg(reason));
        m_btnMarathonContinue->show();
        m_btnMarathonPaste->hide();
        m_marathonBar->show();
        appendThinking("⚠️ Kode terdeteksi belum selesai: " + reason, "#ff8c00");
    } else {
        // Kode lengkap
        if (m_marathonActive) {
            m_marathonStatus->setText("✅ <b>Marathon Selesai:</b> Kode telah lengkap.");
            m_btnMarathonContinue->hide();
            m_btnMarathonPaste->show();
            m_marathonBar->show();
            m_marathonActive = false;
        } else {
            // Tampilkan bar paste jika ada code block
            m_marathonStatus->setText("📋 <b>Moko Code:</b> Kode siap ditempel ke editor.");
            m_btnMarathonContinue->hide();
            m_btnMarathonPaste->show();
            m_marathonBar->show();
        }
    }
}

void ChatWidget::continueMarathon() {
    m_marathonActive = true;
    m_marathonPasses++;
    m_marathonBar->hide();

    startThinking(QString("Melanjutkan kode (Pass %1/5)...").arg(m_marathonPasses));
    appendThinking("→ Memanggil endpoint /v1/marathon/continue (SSE)", "#cc44ff");

    QUrl url("http://127.0.0.1:11435/v1/marathon/continue");
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonObject payload;
    payload["original_prompt"] = m_marathonOriginalPrompt;
    payload["accumulated_code"] = m_marathonAccumulatedCode;
    payload["language"] = m_marathonLanguage;
    payload["reason"] = m_marathonStatus->text();

    m_activeReply = m_networkManager->post(req, QJsonDocument(payload).toJson());
    m_isStreaming = true;
    m_streamBuffer.clear();

    connect(m_activeReply, &QNetworkReply::readyRead, this, &ChatWidget::onMarathonReadyRead);
    connect(m_activeReply, &QNetworkReply::finished,  this, &ChatWidget::onMarathonFinished);
}

void ChatWidget::onMarathonReadyRead() {
    if (!m_activeReply) return;
    m_streamBuffer += m_activeReply->readAll();

    while (true) {
        int nl = m_streamBuffer.indexOf('\n');
        if (nl == -1) break;
        QByteArray line = m_streamBuffer.left(nl).trimmed();
        m_streamBuffer.remove(0, nl + 1);

        if (line.isEmpty() || line == "data: [DONE]") continue;
        if (!line.startsWith("data: ")) continue;

        QByteArray jsonPart = line.mid(6);
        QJsonDocument chunkDoc = QJsonDocument::fromJson(jsonPart);
        if (chunkDoc.isNull() || !chunkDoc.isObject()) continue;
        QJsonObject obj = chunkDoc.object();

        QJsonArray choices = obj["choices"].toArray();
        if (choices.isEmpty()) continue;

        QJsonObject delta = choices[0].toObject()["delta"].toObject();
        if (delta.contains("content")) {
            QString token = delta["content"].toString();
            m_chatHistory->moveCursor(QTextCursor::End);
            m_chatHistory->insertPlainText(token);
            m_chatHistory->verticalScrollBar()->setValue(m_chatHistory->verticalScrollBar()->maximum());

            if (!m_isLoadingSession && !m_currentMessages.isEmpty()) {
                m_currentMessages.last().text += token;
            }
        }
    }
}

void ChatWidget::onMarathonFinished() {
    m_activeReply->deleteLater();
    m_activeReply = nullptr;
    m_isStreaming = false;
    endThinking();

    appendThinking("✓ Lanjutan kode berhasil disambung.", "#00ff88");
    saveCurrentSession();

    // Analisis lagi apakah sudah lengkap atau masih kurang
    if (m_marathonPasses < 5) {
        analyzeCompleteness();
    } else {
        m_marathonStatus->setText("⚠️ <b>Batas Pass Tercapai:</b> Marathon terhenti (max 5 pass).");
        m_btnMarathonContinue->hide();
        m_btnMarathonPaste->show();
        m_marathonBar->show();
        m_marathonActive = false;
    }
}

void ChatWidget::pasteToEditor() {
    emit insertCodeToEditor(m_marathonAccumulatedCode, m_marathonLanguage);
    m_marathonBar->hide();
}

void ChatWidget::stopMarathon() {
    m_marathonActive = false;
    m_marathonPasses = 0;
    m_marathonBar->hide();
}

void ChatWidget::receiveEditorSnapshot(const QString& code, const QString& lang, const QString& filePath) {
    setCodeContext(code);
}

