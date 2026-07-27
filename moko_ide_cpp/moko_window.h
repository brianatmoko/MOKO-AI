#ifndef MOKO_WINDOW_H
#define MOKO_WINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTreeView>
#include <QFileSystemModel>
#include <QTabWidget>
#include <QTabBar>
#include <QToolBar>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QTimer>
#include <QMap>
#include <QListWidget>
#include <QTextBrowser>
#include <QFrame>
#include <QStackedWidget>

#include "code_editor.h"
#include "find_bar.h"
#include "terminal_widget.h"
#include "chat_widget.h"
#include "helper_engine.h"

// ─── MokoWindow ──────────────────────────────────────────────────────────────
class MokoWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MokoWindow(QWidget* parent = nullptr);
    ~MokoWindow();

private slots:
    void onFileActivated(const QModelIndex& index);
    void onFileSingleClicked(const QModelIndex& index);
    void onEditorTabChanged(int index);
    void onTabCloseRequested(int index);
    void onEditorModified(bool modified);
    void saveCurrentFile();
    void toggleTerminal();
    void toggleSidebar();
    void toggleChat();
    void openFolder();
    void showCommandPalette();
    void showFindBar();
    void updateHardwareStats();
    void onBottomTabChanged(int index);
    void openSettings();
    void onInsertCodeToEditor(const QString& code, const QString& lang);
    void newFile();
    void closeCurrentTab();
    void showGitPanel();
    void toggleWordWrap();
    void toggleMinimap();

private:
    // ── Setup helpers ──────────────────────────────────────────────
    void setupMenuBar();
    void setupLayout();
    void setupActivityBar();
    void setupSidebar();
    void setupEditorArea();
    void setupBottomPanel();
    void setupChatPanel();
    void setupStatusBar();
    void applyTheme();

    // ── File helpers ───────────────────────────────────────────────
    void openFile(const QString& path, bool preview = false);
    void setCwd(const QString& path);
    void rebuildTabIndexMap();
    CodeEditor* currentEditor();
    void runHelper();
    void updateBreadcrumb(const QString& path);
    void updateOpenEditorsList();
    void addTerminalTab();
    QLabel* mkSep(QWidget* parent);
    QPushButton* makeActivityBtn(const QString& icon, const QString& tip);
    QWidget* makeSectionHeader(const QString& title, QWidget* content);
    QWidget* createWelcomeWidget();

    // ── Activity Bar ───────────────────────────────────────────────
    QWidget*      m_activityBar   = nullptr;
    QPushButton*  m_btnExplorer   = nullptr;
    QPushButton*  m_btnSearch     = nullptr;
    QPushButton*  m_btnGit        = nullptr;
    QPushButton*  m_btnExtensions = nullptr;
    QPushButton*  m_btnChat       = nullptr;
    QPushButton*  m_btnSettings   = nullptr;

    // ── Sidebar ────────────────────────────────────────────────────
    QWidget*          m_sidebarPanel      = nullptr;
    QWidget*          m_openEditorsSection= nullptr;
    QListWidget*      m_openEditorsList   = nullptr;
    QTreeView*        m_fileTree          = nullptr;
    QFileSystemModel* m_fsModel           = nullptr;
    QLineEdit*        m_searchBox         = nullptr;
    QWidget*          m_explorerSection   = nullptr;

    // ── Splitters ──────────────────────────────────────────────────
    QSplitter* m_mainSplitter   = nullptr;
    QSplitter* m_centerSplitter = nullptr;

    // ── Editor tabs ────────────────────────────────────────────────
    QWidget*    m_editorContainer  = nullptr;
    QTabWidget* m_editorTabs       = nullptr;
    QLabel*     m_breadcrumb       = nullptr;
    FindBar*    m_findBar          = nullptr;
    bool        m_wordWrap         = false;
    bool        m_minimapVisible   = true;

    // Preview tab tracking
    int         m_previewTabIndex  = -1;
    QString     m_previewTabPath;

    // ── Bottom panel ───────────────────────────────────────────────
    QWidget*      m_bottomPanel   = nullptr;
    QTabWidget*   m_bottomTabs    = nullptr;
    TerminalWidget* m_terminal    = nullptr;
    QTextBrowser* m_problemsView  = nullptr;
    QTextBrowser* m_outputView    = nullptr;
    QTextBrowser* m_debugView     = nullptr;
    bool          m_terminalVisible = true;
    int           m_terminalCount   = 1;

    // ── Chat panel ─────────────────────────────────────────────────
    ChatWidget* m_chat = nullptr;

    // ── Status bar labels ──────────────────────────────────────────
    QLabel* m_lblBranch   = nullptr;
    QLabel* m_lblHelper   = nullptr;
    QLabel* m_lblLinCol   = nullptr;
    QLabel* m_lblLang     = nullptr;
    QLabel* m_lblIndent   = nullptr;
    QLabel* m_lblEncoding = nullptr;
    QLabel* m_lblEol      = nullptr;
    QLabel* m_lblCpu      = nullptr;
    QLabel* m_lblRam      = nullptr;
    QLabel* m_lblSaved    = nullptr;
    QLabel* m_lblAiStatus = nullptr;
    QLabel* m_lblWordWrap = nullptr;

    // ── Jantung Helper engine ──────────────────────────────────────
    HelperEngine m_helper;
    QVector<DiagnosticIssue> m_lastIssues;

    // ── Tab tracking: filePath → tab index ────────────────────────
    QMap<QString, int> m_openTabs;

    QString m_workspaceRoot;
    QTimer* m_hwTimer = nullptr;
};

#endif // MOKO_WINDOW_H
