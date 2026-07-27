#ifndef CHAT_WIDGET_H
#define CHAT_WIDGET_H

#include <QWidget>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QListWidget>
#include <QSplitter>
#include <QLabel>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QList>
#include <QTimer>

struct ChatMessage {
    QString sender;
    QString text;
    qint64 ts;
};

class ChatWidget : public QWidget {
    Q_OBJECT
public:
    explicit ChatWidget(QWidget* parent = nullptr);
    ~ChatWidget();

    void appendMessage(const QString& sender, const QString& text);
    void setCodeContext(const QString& context);

    // ── Session Management API ─────────────────
    void newSession();
    void loadSessionList();
    void loadSession(const QString& sessionId);
    void saveCurrentSession();
    void deleteSession(const QString& sessionId);

    // ── X-Ray Thinking Panel API ───────────────
    void startThinking(const QString& status = "Menganalisis...");
    void appendThinking(const QString& text, const QString& color = "");
    void endThinking();

signals:
    void promptSent(const QString& prompt);
    void insertCodeToEditor(const QString& code, const QString& lang);
    void codeEditorContentRequested();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private slots:
    void sendChat();
    void handleNetworkReply(QNetworkReply* reply);
    void onReadyRead();
    void onFinished();
    void onSessionSelected(QListWidgetItem* item);
    void showSessionContextMenu(const QPoint& pos);

    // Slash command popup
    void onInputTextChanged(const QString& text);
    void onSlashItemClicked(QListWidgetItem* item);

    // History dropdown
    void showHistoryMenu();

    // Thinking panel
    void toggleThinkingContent();

    // Marathon slots
    void analyzeCompleteness();
    void handleMarathonAnalysis(QNetworkReply* reply);
    void continueMarathon();
    void onMarathonReadyRead();
    void onMarathonFinished();
    void pasteToEditor();
    void stopMarathon();

public slots:
    void receiveEditorSnapshot(const QString& code, const QString& lang, const QString& filePath);

private:
    void showSlashPopup(const QVector<QPair<QString,QString>>& matches);
    void hideSlashPopup();
    void setupMarathonBar();

    QListWidget* m_sessionList;   // hidden helper (for session load/save logic)
    QPushButton* m_btnHistory = nullptr;  // History dropdown button

    // ── Chat main area ─────────────────────────
    QWidget*     m_chatContainer;
    QTextEdit*   m_chatHistory;
    QLineEdit*   m_inputField;
    QPushButton* m_sendButton;
    QCheckBox*   m_chkAgentMode;

    // ── X-Ray Thinking Panel ───────────────────
    QWidget*     m_thinkingWidget;
    QTextEdit*   m_thinkingBox;
    QLabel*      m_thinkStatus;
    QPushButton* m_thinkToggle;
    bool         m_thinkingActive = false;

    // ── Slash Autocomplete Popup ───────────────
    QListWidget* m_slashPopup;

    // ── Marathon Auto-Continue UI ──────────────
    QWidget*     m_marathonBar = nullptr;
    QLabel*      m_marathonStatus = nullptr;
    QPushButton* m_btnMarathonContinue = nullptr;
    QPushButton* m_btnMarathonPaste = nullptr;
    QPushButton* m_btnMarathonStop = nullptr;

    QNetworkAccessManager* m_networkManager;
    QString m_codeContext;

    // ── Multi-Agent streaming state ────────────
    QNetworkReply* m_activeReply    = nullptr;
    QString        m_currentSender;
    bool           m_isStreaming    = false;
    QByteArray     m_streamBuffer;

    // ── Marathon State ─────────────────────────
    bool           m_marathonActive = false;
    QString        m_marathonAccumulatedCode;
    QString        m_marathonOriginalPrompt;
    QString        m_marathonLanguage;
    int            m_marathonPasses = 0;

    // ── Chat Session State ─────────────────────
    QString            m_currentSessionId;
    QList<ChatMessage> m_currentMessages;
    bool               m_isLoadingSession = false;
};

#endif // CHAT_WIDGET_H
