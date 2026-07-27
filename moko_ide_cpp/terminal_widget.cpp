#include "terminal_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollBar>
#include <QEvent>
#include <QKeyEvent>
#include <QDir>
#include <QFont>
#include <QFontMetrics>
#include <QTextCursor>
#include <QTextBlock>
#include <unistd.h>
#include <fcntl.h>
#include <pty.h>
#include <utmp.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <cerrno>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
TerminalWidget::TerminalWidget(QWidget* parent)
    : QWidget(parent)
    , m_cwd(QDir::homePath())
{
    setStyleSheet("background:#1e1e1e;");

    QVBoxLayout* vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(0, 0, 0, 0);
    vlay->setSpacing(0);

    // ── Header bar ────────────────────────────────────────────────
    QWidget* hdr = new QWidget(this);
    hdr->setFixedHeight(28);
    hdr->setStyleSheet("background:#252526; border-bottom:1px solid #1e1e1e;");
    QHBoxLayout* hlay = new QHBoxLayout(hdr);
    hlay->setContentsMargins(10, 0, 6, 0);
    hlay->setSpacing(8);

    QLabel* title = new QLabel("TERMINAL", hdr);
    title->setStyleSheet("color:#858585; font-size:11px; font-weight:700; letter-spacing:1.2px;");
    hlay->addWidget(title);

    m_cwdLabel = new QLabel(m_cwd, hdr);
    m_cwdLabel->setStyleSheet("color:#444; font-size:10px; font-family:'Fira Code',monospace;");
    hlay->addWidget(m_cwdLabel, 1);

    QPushButton* btnClear = new QPushButton("⊘ Clear", hdr);
    btnClear->setFixedHeight(20);
    btnClear->setStyleSheet(
        "QPushButton { background:transparent; border:1px solid #3c3c3c; color:#777; "
        "font-size:10px; padding:0 8px; border-radius:3px; }"
        "QPushButton:hover { color:#ccc; border-color:#555; }"
    );
    connect(btnClear, &QPushButton::clicked, this, &TerminalWidget::clearTerminal);
    hlay->addWidget(btnClear);

    QPushButton* btnRestart = new QPushButton("↺", hdr);
    btnRestart->setFixedSize(24, 20);
    btnRestart->setToolTip("Restart shell");
    btnRestart->setStyleSheet(
        "QPushButton { background:transparent; border:1px solid #3c3c3c; color:#777; "
        "font-size:13px; border-radius:3px; }"
        "QPushButton:hover { color:#ccc; border-color:#555; }"
    );
    connect(btnRestart, &QPushButton::clicked, this, &TerminalWidget::restartShell);
    hlay->addWidget(btnRestart);
    vlay->addWidget(hdr);

    // ── Output area (QPlainTextEdit — plain text, not HTML) ───────
    m_output = new QPlainTextEdit(this);
    m_output->setReadOnly(true);
    m_output->setUndoRedoEnabled(false);
    m_output->setWordWrapMode(QTextOption::WrapAnywhere);
    m_output->setMaximumBlockCount(8000);  // limit history

    QFont f("Fira Code", 11);
    f.setStyleHint(QFont::Monospace);
    f.setFixedPitch(true);
    m_output->setFont(f);
    m_output->setStyleSheet(
        "QPlainTextEdit { background:#1e1e1e; color:#cccccc; border:none; "
        "padding:6px 10px; font-family:'Fira Code','Consolas',monospace; font-size:11px; }"
        "QScrollBar:vertical { background:#1e1e1e; width:8px; }"
        "QScrollBar::handle:vertical { background:#3c3c3c; border-radius:4px; }"
    );
    vlay->addWidget(m_output, 1);

    // ── Input row ─────────────────────────────────────────────────
    QWidget* inputWidget = new QWidget(this);
    inputWidget->setFixedHeight(34);
    inputWidget->setStyleSheet("background:#252526; border-top:1px solid #1e1e1e;");
    QHBoxLayout* ilay = new QHBoxLayout(inputWidget);
    ilay->setContentsMargins(8, 4, 8, 4);
    ilay->setSpacing(6);

    QLabel* prompt = new QLabel("❯", inputWidget);
    prompt->setStyleSheet("color:#4ec9b0; font-size:14px; font-weight:bold;");
    ilay->addWidget(prompt);

    m_input = new QLineEdit(inputWidget);
    m_input->setPlaceholderText("Type command here…");
    m_input->setFont(f);
    m_input->setStyleSheet(
        "QLineEdit { background:transparent; border:none; color:#d4d4d4; "
        "font-family:'Fira Code',monospace; font-size:11px; }"
    );
    m_input->installEventFilter(this);
    connect(m_input, &QLineEdit::returnPressed, this, &TerminalWidget::handleInput);
    ilay->addWidget(m_input, 1);
    vlay->addWidget(inputWidget);

    // ── Signal for cross-thread PTY output ────────────────────────
    connect(this, &TerminalWidget::rawDataReceived,
            this, &TerminalWidget::onRawData,
            Qt::QueuedConnection);

    // Start the shell
    startShell();
}

TerminalWidget::~TerminalWidget() {
    killShell();
}

// ─── Start shell ─────────────────────────────────────────────────────────────
void TerminalWidget::startShell() {
    m_running = true;

    // Set terminal window size
    struct winsize ws;
    ws.ws_col = 220;
    ws.ws_row = 50;
    ws.ws_xpixel = 0;
    ws.ws_ypixel = 0;

    m_childPid = forkpty(&m_masterFd, nullptr, nullptr, &ws);

    if (m_childPid < 0) {
        m_output->appendPlainText("[ERROR] forkpty failed: " + QString(strerror(errno)));
        return;
    }

    if (m_childPid == 0) {
        // ── CHILD PROCESS ──────────────────────────────────────────
        // Change to workspace directory
        if (!m_cwd.isEmpty())
            chdir(m_cwd.toUtf8().constData());

        // Set essential env vars
        setenv("TERM", "xterm-256color", 1);
        setenv("COLORTERM", "truecolor", 1);
        setenv("PAGER", "cat", 1);
        setenv("LESS", "-RF", 1);
        unsetenv("PYTHONDONTWRITEBYTECODE");

        // Run bash WITH .bashrc so virtualenvs, aliases, PS1 all work
        execl("/bin/bash", "/bin/bash", "--login", "-i", nullptr);

        // fallback
        execl("/bin/sh", "/bin/sh", nullptr);
        _exit(1);
    }

    // ── PARENT PROCESS ─────────────────────────────────────────────
    // Non-blocking reads
    int flags = fcntl(m_masterFd, F_GETFL, 0);
    fcntl(m_masterFd, F_SETFL, flags | O_NONBLOCK);

    // Start reader thread
    m_readThread = std::thread(&TerminalWidget::readLoop, this);
}

// ─── Read loop (background thread) ───────────────────────────────────────────
void TerminalWidget::readLoop() {
    char buf[4096];
    while (m_running) {
        ssize_t n = read(m_masterFd, buf, sizeof(buf));
        if (n > 0) {
            emit rawDataReceived(QByteArray(buf, (int)n));
        } else if (n == 0) {
            m_running = false;
            break;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
            } else {
                m_running = false;
                break;
            }
        }
    }
}

// ─── Process raw PTY bytes ────────────────────────────────────────────────────
// This strips ANSI escape sequences and handles \r\n and \b correctly,
// then appends to the QPlainTextEdit on the GUI thread.
void TerminalWidget::onRawData(const QByteArray& data) {
    processBytes(data);
}

void TerminalWidget::processBytes(const QByteArray& raw) {
    // We maintain m_lineBuf as the "current line being built"
    // We only append full lines or flush when we encounter output.

    QString text = QString::fromLocal8Bit(raw);

    // Strip ANSI escape sequences: ESC[ ... (letter)
    // We do a simple state machine strip
    QString stripped;
    stripped.reserve(text.size());
    bool inEsc = false;
    bool inOsc = false;
    for (int i = 0; i < text.size(); ++i) {
        QChar c = text[i];
        if (inEsc) {
            if (c == '[') {
                // CSI sequence — keep reading until letter
                i++;
                while (i < text.size() && !text[i].isLetter()) i++;
                inEsc = false;
            } else if (c == ']') {
                inOsc = true;
                inEsc = false;
            } else if (c.isLetter() || c == '(' || c == ')' || c == '=') {
                inEsc = false;  // single-char escape
            } else {
                inEsc = false;
            }
        } else if (inOsc) {
            // OSC sequence ends with BEL or ST
            if (c == '\007' || (c == '\\' && i > 0 && text[i-1] == '\033')) {
                inOsc = false;
            }
        } else if (c == '\033') {
            inEsc = true;
        } else if (c == '\007') {
            // BEL — ignore
        } else if (c == '\010') {
            // Backspace — remove last char from lineBuf
            if (!m_lineBuf.isEmpty()) m_lineBuf.chop(1);
        } else if (c == '\r') {
            // Carriage return — go to beginning of current line
            // We'll flush what we have and mark CR
            m_lineBuf.clear();
        } else {
            stripped.append(c);
        }
    }

    // Now process stripped text char by char for newlines
    for (int i = 0; i < stripped.size(); ++i) {
        QChar c = stripped[i];
        if (c == '\n') {
            // Append current line buffer as a new block
            QTextCursor cur = m_output->textCursor();
            cur.movePosition(QTextCursor::End);
            if (!m_lineBuf.isEmpty() || true) {
                // Append the accumulated line
                cur.insertText(m_lineBuf + "\n");
            }
            m_lineBuf.clear();
        } else {
            m_lineBuf.append(c);
        }
    }

    // Flush partial line (prompt text without newline)
    if (!m_lineBuf.isEmpty()) {
        // Replace the last block with the current line buffer
        QTextCursor cur = m_output->textCursor();
        cur.movePosition(QTextCursor::End);
        cur.select(QTextCursor::LineUnderCursor);
        cur.removeSelectedText();
        cur.insertText(m_lineBuf);
    }

    // Keep scroll at bottom
    QScrollBar* sb = m_output->verticalScrollBar();
    sb->setValue(sb->maximum());
}

// ─── Write to PTY ────────────────────────────────────────────────────────────
void TerminalWidget::writeToPTY(const std::string& data) {
    if (m_masterFd >= 0) {
        ssize_t written = 0;
        ssize_t total = (ssize_t)data.size();
        const char* ptr = data.data();
        while (written < total) {
            ssize_t n = write(m_masterFd, ptr + written, total - written);
            if (n <= 0) break;
            written += n;
        }
    }
}

// ─── Handle user input ────────────────────────────────────────────────────────
void TerminalWidget::handleInput() {
    QString cmd = m_input->text().trimmed();
    m_input->clear();

    // History management
    if (!cmd.isEmpty()) {
        m_history.removeAll(cmd);   // de-dup
        m_history.append(cmd);
        if (m_history.size() > 500) m_history.removeFirst();
    }
    m_historyIdx = m_history.size();
    m_savedInput.clear();

    // Send to PTY (with newline)
    writeToPTY(cmd.toLocal8Bit().toStdString() + "\n");
}

// ─── Clear / Kill / Restart ───────────────────────────────────────────────────
void TerminalWidget::clearTerminal() {
    m_output->clear();
    m_lineBuf.clear();
}

void TerminalWidget::killShell() {
    m_running = false;
    if (m_masterFd >= 0) {
        ::close(m_masterFd);
        m_masterFd = -1;
    }
    if (m_childPid > 0) {
        ::kill(m_childPid, SIGKILL);
        int status;
        ::waitpid(m_childPid, &status, WNOHANG);
        m_childPid = -1;
    }
    if (m_readThread.joinable()) {
        m_readThread.join();
    }
}

void TerminalWidget::restartShell() {
    killShell();
    m_output->clear();
    m_lineBuf.clear();
    m_running = true;
    startShell();
}

// ─── setCwd: only change dir for NEW shells; do NOT send cd to running shell ──
void TerminalWidget::setCwd(const QString& path) {
    m_cwd = path;
    if (m_cwdLabel) m_cwdLabel->setText(path);
    // Do NOT send "cd" to a running shell — user controls their own shell.
    // setCwd() only sets the default dir for the NEXT shell restart.
}

void TerminalWidget::executeCommand(const QString& cmd) {
    m_input->setText(cmd);
    handleInput();
}

// ─── Keyboard: history navigation ────────────────────────────────────────────
bool TerminalWidget::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_input && event->type() == QEvent::KeyPress) {
        QKeyEvent* ke = static_cast<QKeyEvent*>(event);
        switch (ke->key()) {
        case Qt::Key_Up:
            if (m_historyIdx > 0) {
                if (m_historyIdx == m_history.size())
                    m_savedInput = m_input->text();
                m_historyIdx--;
                m_input->setText(m_history[m_historyIdx]);
            }
            return true;
        case Qt::Key_Down:
            if (m_historyIdx < m_history.size() - 1) {
                m_historyIdx++;
                m_input->setText(m_history[m_historyIdx]);
            } else if (m_historyIdx == m_history.size() - 1) {
                m_historyIdx++;
                m_input->setText(m_savedInput);
            }
            return true;
        case Qt::Key_C:
            // Ctrl+C → send SIGINT to shell via PTY
            if (ke->modifiers() & Qt::ControlModifier) {
                writeToPTY("\x03");
                m_input->clear();
                return true;
            }
            break;
        case Qt::Key_L:
            // Ctrl+L → clear
            if (ke->modifiers() & Qt::ControlModifier) {
                clearTerminal();
                return true;
            }
            break;
        case Qt::Key_D:
            // Ctrl+D → EOF
            if (ke->modifiers() & Qt::ControlModifier) {
                writeToPTY("\x04");
                return true;
            }
            break;
        default:
            break;
        }
    }
    return QWidget::eventFilter(watched, event);
}
