#ifndef TERMINAL_WIDGET_H
#define TERMINAL_WIDGET_H

#include <QWidget>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QString>
#include <thread>
#include <atomic>

class TerminalWidget : public QWidget {
    Q_OBJECT
public:
    explicit TerminalWidget(QWidget* parent = nullptr);
    ~TerminalWidget();

    void setCwd(const QString& path);
    void executeCommand(const QString& cmd);

signals:
    void rawDataReceived(const QByteArray& data);

private slots:
    void handleInput();
    void onRawData(const QByteArray& data);
    void clearTerminal();
    void killShell();
    void restartShell();

private:
    void startShell();
    void readLoop();
    void writeToPTY(const std::string& data);
    void processBytes(const QByteArray& data);

    // ── UI ─────────────────────────────────────
    QPlainTextEdit* m_output;    // raw text output (no HTML)
    QLineEdit*      m_input;
    QLabel*         m_cwdLabel;

    // ── PTY state ──────────────────────────────
    int             m_masterFd  = -1;
    pid_t           m_childPid  = -1;
    std::thread     m_readThread;
    std::atomic<bool> m_running{false};
    QString         m_cwd;

    // ── Line buffer for carriage-return handling ─
    QString m_lineBuf;

    // ── Command history ─────────────────────────
    QStringList m_history;
    int         m_historyIdx = -1;
    QString     m_savedInput;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
};

#endif // TERMINAL_WIDGET_H
