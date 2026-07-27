#include "moko_window.h"
#include "syntax_highlighter.h"
#include "settings_dialog.h"

#include <QApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QShortcut>
#include <QKeySequence>
#include <QHeaderView>
#include <QScrollBar>
#include <QFont>
#include <QFontMetrics>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDialog>
#include <QListWidget>
#include <QListWidgetItem>
#include <QDirIterator>
#include <QPainter>
#include <QFrame>
#include <QTextCursor>
#include <QStatusBar>
#include <QGraphicsDropShadowEffect>
#include <QPropertyAnimation>
#include <QScrollArea>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static QString langFromPath(const QString& path) {
    const QString ext = QFileInfo(path).suffix().toLower();
    if (ext == "py")                                   return "Python";
    if (ext == "cpp" || ext == "cxx" || ext == "cc")  return "C++";
    if (ext == "h"   || ext == "hpp")                  return "C++ Header";
    if (ext == "c")                                    return "C";
    if (ext == "js")                                   return "JavaScript";
    if (ext == "ts")                                   return "TypeScript";
    if (ext == "md")                                   return "Markdown";
    if (ext == "sh"  || ext == "bash")                 return "Shell";
    if (ext == "json")                                 return "JSON";
    if (ext == "yaml" || ext == "yml")                 return "YAML";
    if (ext == "html" || ext == "htm")                 return "HTML";
    if (ext == "css")                                  return "CSS";
    if (ext == "rs")                                   return "Rust";
    if (ext == "go")                                   return "Go";
    if (ext == "java")                                 return "Java";
    if (ext == "kt")                                   return "Kotlin";
    if (ext == "txt")                                  return "Plain Text";
    if (ext == "toml")                                 return "TOML";
    if (ext == "xml")                                  return "XML";
    if (ext == "cmake" || ext == "")                   return "CMake";
    return "Text";
}

static QString highlightLang(const QString& displayLang) {
    if (displayLang == "Python")                return "python";
    if (displayLang.startsWith("C++"))          return "cpp";
    if (displayLang == "C")                     return "c";
    if (displayLang == "JavaScript")            return "javascript";
    if (displayLang == "TypeScript")            return "typescript";
    if (displayLang == "Shell")                 return "shell";
    if (displayLang == "HTML")                  return "html";
    if (displayLang == "CSS")                   return "css";
    if (displayLang == "JSON")                  return "json";
    return "";
}

static QString iconForExtension(const QString& ext) {
    if (ext == "py")                    return "🐍";
    if (ext == "cpp" || ext == "cxx")   return "⚙";
    if (ext == "h" || ext == "hpp")     return "📋";
    if (ext == "js" || ext == "ts")     return "🟨";
    if (ext == "json")                  return "{}";
    if (ext == "md")                    return "📝";
    if (ext == "html")                  return "🌐";
    if (ext == "css")                   return "🎨";
    if (ext == "sh")                    return "💲";
    if (ext == "yaml" || ext == "yml")  return "⚡";
    if (ext == "rs")                    return "🦀";
    if (ext == "go")                    return "🐹";
    return "📄";
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
MokoWindow::MokoWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_workspaceRoot(QDir::homePath())
{
    setWindowTitle("Moko IDE");
    resize(1760, 1000);
    applyTheme();
    setupMenuBar();
    setupLayout();
    setupStatusBar();

    m_hwTimer = new QTimer(this);
    connect(m_hwTimer, &QTimer::timeout, this, &MokoWindow::updateHardwareStats);
    m_hwTimer->start(3000);

    // ── Keyboard Shortcuts (VSCode-identical) ──────────────────────
    new QShortcut(QKeySequence("Ctrl+S"),          this, SLOT(saveCurrentFile()));
    new QShortcut(QKeySequence("Ctrl+N"),          this, SLOT(newFile()));
    new QShortcut(QKeySequence("Ctrl+W"),          this, SLOT(closeCurrentTab()));
    new QShortcut(QKeySequence("Ctrl+`"),          this, SLOT(toggleTerminal()));
    new QShortcut(QKeySequence("Ctrl+Shift+E"),    this, SLOT(toggleSidebar()));
    new QShortcut(QKeySequence("Ctrl+Shift+G"),    this, SLOT(toggleChat()));
    new QShortcut(QKeySequence("Ctrl+Shift+P"),    this, SLOT(showCommandPalette()));
    new QShortcut(QKeySequence("Ctrl+P"),          this, SLOT(showCommandPalette()));
    new QShortcut(QKeySequence("Ctrl+F"),          this, SLOT(showFindBar()));
    new QShortcut(QKeySequence("Ctrl+K, Ctrl+O"), this, SLOT(openFolder()));
    new QShortcut(QKeySequence("Ctrl+,"),          this, SLOT(openSettings()));
    new QShortcut(QKeySequence("Ctrl+Z"),          this, [this]{ if(auto*e=currentEditor())e->undo(); });
    new QShortcut(QKeySequence("Ctrl+Y"),          this, [this]{ if(auto*e=currentEditor())e->redo(); });

    setCwd(m_workspaceRoot);
    updateHardwareStats();
}

MokoWindow::~MokoWindow() {}

// ─────────────────────────────────────────────────────────────────────────────
// Theme — VSCode Dark+ faithful QSS
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::applyTheme() {
    setStyleSheet(R"(
/* ── Base ──────────────────────────────────────────────────────────── */
* {
    font-family: 'Inter', 'Segoe UI', 'Noto Sans', 'Ubuntu', sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget {
    background: #0a0a0a;
    color: #c0c0c0;
    outline: none;
}

/* ── Menu Bar ───────────────────────────────────────────────────────── */
QMenuBar {
    background: #0d0d0d;
    color: #b0b0b0;
    border-bottom: 1px solid #1a1a1a;
    padding: 1px 0;
}
QMenuBar::item {
    padding: 4px 10px;
    background: transparent;
}
QMenuBar::item:selected {
    background: #1a1a1a;
    color: #ff6b35;
}
QMenuBar::item:pressed {
    background: #111111;
}
QMenu {
    background: #0d0d0d;
    color: #c0c0c0;
    border: 1px solid #222222;
    border-radius: 6px;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 32px 6px 20px;
    min-width: 180px;
}
QMenu::item:selected {
    background: rgba(255,107,53,0.18);
    color: #ff6b35;
}
QMenu::item:disabled {
    color: #444444;
}
QMenu::separator {
    height: 1px;
    background: #1a1a1a;
    margin: 3px 8px;
}
QMenu::indicator {
    width: 16px;
    height: 16px;
}

/* ── Splitter ───────────────────────────────────────────────────────── */
QSplitter::handle {
    background: #111111;
}
QSplitter::handle:hover {
    background: #ff6b35;
}
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }

/* ── Tab Widget ─────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: none;
    background: #0a0a0a;
    margin: 0;
    padding: 0;
}

/* ── Editor Tab Bar ─────────────────────────────────────────────────── */
QTabBar {
    background: #0d0d0d;
    border: none;
}
QTabBar::tab {
    background: #111111;
    color: #666666;
    padding: 7px 18px 7px 14px;
    border: none;
    border-right: 1px solid #0a0a0a;
    font-size: 12px;
    min-width: 80px;
    max-width: 200px;
}
QTabBar::tab:selected {
    background: #0a0a0a;
    color: #e0e0e0;
    border-top: 2px solid #ff6b35;
}
QTabBar::tab:hover:!selected {
    background: #141414;
    color: #999999;
}
QTabBar::close-button {
    subcontrol-position: right;
}

/* ── Bottom Tab Bar ─────────────────────────────────────────────────── */
QTabWidget#bottomTabs::pane {
    border: none;
    border-top: 1px solid #1a1a1a;
}
QTabWidget#bottomTabs QTabBar {
    background: #0d0d0d;
}
QTabWidget#bottomTabs QTabBar::tab {
    padding: 5px 16px;
    font-size: 11px;
    font-weight: 600;
    min-width: 60px;
    background: #0d0d0d;
    color: #555555;
    border: none;
}
QTabWidget#bottomTabs QTabBar::tab:selected {
    background: #0a0a0a;
    color: #e0e0e0;
    border-top: 2px solid #ff6b35;
}
QTabWidget#bottomTabs QTabBar::tab:hover:!selected {
    color: #999999;
    background: #111111;
}

/* ── Code Editor / QPlainTextEdit ───────────────────────────────────── */
QPlainTextEdit {
    background: #0a0a0a;
    color: #d0d0d0;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    border: none;
    padding: 0;
    selection-background-color: rgba(255,107,53,0.22);
    selection-color: #ffffff;
}

/* ── Breadcrumb ─────────────────────────────────────────────────────── */
QLabel#breadcrumb {
    color: #666666;
    font-size: 11px;
    padding: 3px 14px;
    background: #0a0a0a;
    border-bottom: 1px solid #141414;
    min-height: 22px;
}

/* ── Sidebar / File Tree ────────────────────────────────────────────── */
QTreeView {
    background: #0d0d0d;
    color: #b0b0b0;
    border: none;
    font-size: 12px;
    show-decoration-selected: 1;
}
QTreeView::item {
    padding: 1px 4px;
    min-height: 22px;
}
QTreeView::item:hover    { background: #141414; }
QTreeView::item:selected { background: rgba(255,107,53,0.18); color: #ff6b35; }
QTreeView::branch {
    background: #0d0d0d;
}

/* ── Open Editors List ──────────────────────────────────────────────── */
QListWidget {
    background: #0d0d0d;
    color: #b0b0b0;
    border: none;
    font-size: 12px;
    outline: none;
}
QListWidget::item {
    padding: 2px 10px 2px 24px;
    min-height: 22px;
}
QListWidget::item:hover    { background: #141414; }
QListWidget::item:selected { background: rgba(255,107,53,0.18); color: #ff6b35; }

/* ── QTextBrowser ───────────────────────────────────────────────────── */
QTextBrowser {
    background: #0a0a0a;
    color: #d0d0d0;
    border: none;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── LineEdit ───────────────────────────────────────────────────────── */
QLineEdit {
    background: #111111;
    color: #c0c0c0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    selection-background-color: rgba(255,107,53,0.25);
}
QLineEdit:focus {
    border: 1px solid #ff6b35;
    outline: none;
}


/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {
    background: transparent;
    color: #b0b0b0;
    border: none;
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 4px;
}
QPushButton:hover  { background: #1a1a1a; color: #e0e0e0; }
QPushButton:pressed { background: rgba(255,107,53,0.20); }
QPushButton:checked { color: #ffffff; }

/* ── Scroll Bars ────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0a0a0a;
    width: 8px;
    margin: 0;
    border: none;
}
QScrollBar:horizontal {
    background: #0a0a0a;
    height: 8px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {
    background: #2a2a2a;
    border-radius: 4px;
    min-height: 28px;
    min-width: 28px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {
    background: #444444;
}
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ── Status Bar ─────────────────────────────────────────────────────── */
QStatusBar {
    background: #0d0d0d;
    color: #888888;
    font-size: 11px;
    border: none;
    border-top: 1px solid #1a1a1a;
    padding: 0;
}
QStatusBar QLabel {
    color: #888888;
    padding: 0 6px;
    font-size: 11px;
    background: transparent;
}
QStatusBar QLabel:hover {
    background: rgba(255,107,53,0.10);
    color: #ff6b35;
}
QStatusBar::item { border: none; }

/* ── CheckBox ───────────────────────────────────────────────────────── */
QCheckBox { color: #888; font-size: 12px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #ff6b35;
    border-color: #ff6b35;
}

/* ── Tooltip ────────────────────────────────────────────────────────── */
QToolTip {
    background: #111111;
    color: #d0d0d0;
    border: 1px solid #2a2a2a;
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 4px;
}

/* ── Dialog ─────────────────────────────────────────────────────────── */
QDialog {
    background: #0d0d0d;
    color: #c0c0c0;
}

/* ── ComboBox ───────────────────────────────────────────────────────── */
QComboBox {
    background: #111111;
    color: #c0c0c0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 3px 8px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #0d0d0d;
    color: #c0c0c0;
    selection-background-color: rgba(255,107,53,0.18);
    border: 1px solid #222222;
}

/* ── Frame separators ───────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #1a1a1a;
}
)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Menu Bar
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupMenuBar() {
    QMenuBar* mb = menuBar();

    // File
    QMenu* mFile = mb->addMenu("File");
    mFile->addAction("New File",         this, &MokoWindow::newFile,           QKeySequence("Ctrl+N"));
    mFile->addAction("Open Folder…",    this, &MokoWindow::openFolder,        QKeySequence("Ctrl+K Ctrl+O"));
    mFile->addSeparator();
    mFile->addAction("Save",             this, &MokoWindow::saveCurrentFile,   QKeySequence("Ctrl+S"));
    mFile->addSeparator();
    mFile->addAction("Settings…",       this, &MokoWindow::openSettings,      QKeySequence("Ctrl+,"));
    mFile->addSeparator();
    mFile->addAction("Close Tab",        this, &MokoWindow::closeCurrentTab,   QKeySequence("Ctrl+W"));
    mFile->addAction("Quit",             qApp, &QApplication::quit,            QKeySequence("Ctrl+Q"));

    // Edit
    QMenu* mEdit = mb->addMenu("Edit");
    mEdit->addAction("Undo",            this, [this]{ if(auto*e=currentEditor())e->undo(); }, QKeySequence("Ctrl+Z"));
    mEdit->addAction("Redo",            this, [this]{ if(auto*e=currentEditor())e->redo(); }, QKeySequence("Ctrl+Y"));
    mEdit->addSeparator();
    mEdit->addAction("Cut",             this, [this]{ if(auto*e=currentEditor())e->cut(); },   QKeySequence("Ctrl+X"));
    mEdit->addAction("Copy",            this, [this]{ if(auto*e=currentEditor())e->copy(); },  QKeySequence("Ctrl+C"));
    mEdit->addAction("Paste",           this, [this]{ if(auto*e=currentEditor())e->paste(); }, QKeySequence("Ctrl+V"));
    mEdit->addSeparator();
    mEdit->addAction("Find / Replace…", this, &MokoWindow::showFindBar,        QKeySequence("Ctrl+F"));
    mEdit->addAction("Select All",      this, [this]{ if(auto*e=currentEditor())e->selectAll(); }, QKeySequence("Ctrl+A"));

    // Selection
    QMenu* mSel = mb->addMenu("Selection");
    mSel->addAction("Select All",       this, [this]{ if(auto*e=currentEditor())e->selectAll(); }, QKeySequence("Ctrl+A"));

    // View
    QMenu* mView = mb->addMenu("View");
    mView->addAction("Command Palette…",this, &MokoWindow::showCommandPalette, QKeySequence("Ctrl+Shift+P"));
    mView->addSeparator();
    mView->addAction("Explorer",        this, &MokoWindow::toggleSidebar,     QKeySequence("Ctrl+Shift+E"));
    mView->addAction("Source Control",  this, &MokoWindow::showGitPanel);
    mView->addAction("AI Chat",         this, &MokoWindow::toggleChat,        QKeySequence("Ctrl+Shift+G"));
    mView->addSeparator();
    mView->addAction("Terminal",        this, &MokoWindow::toggleTerminal,    QKeySequence("Ctrl+`"));
    mView->addSeparator();
    QAction* wrapAct = mView->addAction("Word Wrap", this, &MokoWindow::toggleWordWrap);
    wrapAct->setCheckable(true);
    QAction* mmAct = mView->addAction("Minimap", this, &MokoWindow::toggleMinimap);
    mmAct->setCheckable(true);
    mmAct->setChecked(true);

    // Go
    QMenu* mGo = mb->addMenu("Go");
    mGo->addAction("Go to File…",      this, &MokoWindow::showCommandPalette, QKeySequence("Ctrl+P"));

    // Moko
    QMenu* mMoko = mb->addMenu("Moko");
    mMoko->addAction("Settings (API & Worker Pool)…", this, &MokoWindow::openSettings);
    mMoko->addSeparator();
    mMoko->addAction("About Moko IDE", this, [this]{
        QDialog* d = new QDialog(this);
        d->setWindowTitle("About Moko IDE");
        d->setFixedSize(440, 300);
        d->setStyleSheet("background:#0a0a0a; color:#d0d0d0;");
        QVBoxLayout* lay = new QVBoxLayout(d);
        lay->setAlignment(Qt::AlignCenter);
        QLabel* logo = new QLabel("MOKO", d);
        logo->setAlignment(Qt::AlignCenter);
        logo->setStyleSheet(
            "font-size: 42px; font-weight: 900; color: #ff6b35; "
            "letter-spacing: 8px; font-family: 'Inter','Segoe UI',sans-serif; "
            "border: 2px solid rgba(255,107,53,0.35); border-radius: 10px; "
            "padding: 6px 22px; background: rgba(255,107,53,0.05);"
        );
        QLabel* txt = new QLabel(
            "<h3 style='color:#e0e0e0; margin:6px 0; font-weight:600'>Moko IDE</h3>"
            "<p style='color:#666; margin:0'>Version 8.0 — Native Edition</p>"
            "<p style='color:#555; font-size:12px; margin-top:12px'>"
            "Native C++/Qt5 &nbsp;·&nbsp; AI-Powered &nbsp;·&nbsp; Jantung Helper<br>"
            "Marathon Engine &nbsp;·&nbsp; RAG System &nbsp;·&nbsp; MOKO LLM</p>"
            "<p style='color:#3a3a3a; font-size:11px; margin-top:12px'>© 2026 Brian Atmokoo</p>", d);
        txt->setAlignment(Qt::AlignCenter);
        txt->setTextFormat(Qt::RichText);
        txt->setWordWrap(true);
        lay->addWidget(logo);
        lay->addSpacing(4);
        lay->addWidget(txt);
        QPushButton* ok = new QPushButton("OK", d);
        ok->setFixedWidth(80);
        ok->setStyleSheet(
            "background: rgba(255,107,53,0.15); color:#ff6b35; "
            "border: 1px solid rgba(255,107,53,0.4); border-radius:4px; padding:5px;"
        );
        connect(ok, &QPushButton::clicked, d, &QDialog::accept);
        lay->addWidget(ok, 0, Qt::AlignCenter);
        d->exec();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Layout
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupLayout() {
    QWidget* central = new QWidget(this);
    setCentralWidget(central);
    QHBoxLayout* root = new QHBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    setupActivityBar();
    root->addWidget(m_activityBar);

    m_mainSplitter = new QSplitter(Qt::Horizontal, central);
    m_mainSplitter->setHandleWidth(1);
    root->addWidget(m_mainSplitter, 1);

    setupSidebar();
    setupEditorArea();
    setupChatPanel();

    m_mainSplitter->setSizes({240, 1080, 400});
    m_mainSplitter->setCollapsible(0, true);
    m_mainSplitter->setCollapsible(2, true);
}

// ─────────────────────────────────────────────────────────────────────────────
// Activity Bar Helper
// ─────────────────────────────────────────────────────────────────────────────
QPushButton* MokoWindow::makeActivityBtn(const QString& icon, const QString& tip) {
    QPushButton* b = new QPushButton(icon, m_activityBar);
    b->setToolTip(tip);
    b->setFixedSize(48, 48);
    b->setCheckable(true);
    b->setStyleSheet(R"(
        QPushButton {
            background: transparent;
            border: none;
            font-size: 20px;
            color: #555555;
            border-left: 2px solid transparent;
            padding: 0;
        }
        QPushButton:hover {
            color: #b0b0b0;
            background: rgba(255,107,53,0.06);
        }
        QPushButton:checked {
            color: #e0e0e0;
            border-left: 2px solid #ff6b35;
        }
    )");
    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Activity Bar
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupActivityBar() {
    m_activityBar = new QWidget();
    m_activityBar->setFixedWidth(48);
    m_activityBar->setStyleSheet(
        "background:#080808; border-right: 1px solid #1a1a1a;"
    );
    m_activityBar->setObjectName("activityBar");

    QVBoxLayout* lay = new QVBoxLayout(m_activityBar);
    lay->setContentsMargins(0, 4, 0, 4);
    lay->setSpacing(0);

    // ── Top buttons ────────────────────────────────────────────────
    m_btnExplorer = makeActivityBtn("🗂", "Explorer (Ctrl+Shift+E)");
    m_btnExplorer->setChecked(true);
    connect(m_btnExplorer, &QPushButton::clicked, this, &MokoWindow::toggleSidebar);
    lay->addWidget(m_btnExplorer);

    m_btnSearch = makeActivityBtn("🔍", "Find (Ctrl+F)");
    connect(m_btnSearch, &QPushButton::clicked, this, &MokoWindow::showFindBar);
    lay->addWidget(m_btnSearch);

    m_btnGit = makeActivityBtn("⎇", "Source Control");
    connect(m_btnGit, &QPushButton::clicked, this, &MokoWindow::showGitPanel);
    lay->addWidget(m_btnGit);

    // Open Folder
    QPushButton* btnOpen = makeActivityBtn("📂", "Open Folder (Ctrl+K Ctrl+O)");
    connect(btnOpen, &QPushButton::clicked, this, &MokoWindow::openFolder);
    lay->addWidget(btnOpen);

    // Command Palette
    QPushButton* btnCmd = makeActivityBtn("⌘", "Command Palette (Ctrl+Shift+P)");
    connect(btnCmd, &QPushButton::clicked, this, &MokoWindow::showCommandPalette);
    lay->addWidget(btnCmd);

    lay->addStretch(1);

    // ── Bottom buttons ─────────────────────────────────────────────
    m_btnChat = makeActivityBtn("🤖", "AI Chat (Ctrl+Shift+G)");
    m_btnChat->setChecked(true);
    connect(m_btnChat, &QPushButton::clicked, this, &MokoWindow::toggleChat);
    lay->addWidget(m_btnChat);

    m_btnSettings = makeActivityBtn("⚙", "Settings (Ctrl+,)");
    connect(m_btnSettings, &QPushButton::clicked, this, &MokoWindow::openSettings);
    lay->addWidget(m_btnSettings);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section Header Helper (collapsible)
// ─────────────────────────────────────────────────────────────────────────────
QWidget* MokoWindow::makeSectionHeader(const QString& title, QWidget* content) {
    QPushButton* hdr = new QPushButton("▾  " + title);
    hdr->setCheckable(true);
    hdr->setChecked(true);
    hdr->setStyleSheet(R"(
        QPushButton {
            background: #0d0d0d;
            color: #888888;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-align: left;
            padding: 6px 10px;
            border: none;
            border-bottom: 1px solid #111111;
        }
        QPushButton:hover {
            background: #111111;
            color: #ff6b35;
        }
    )");
    connect(hdr, &QPushButton::toggled, [hdr, content](bool checked){
        content->setVisible(checked);
        hdr->setText((checked ? "▾  " : "▸  ") + hdr->text().mid(3));
    });
    return hdr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupSidebar() {
    m_sidebarPanel = new QWidget(m_mainSplitter);
    m_sidebarPanel->setMinimumWidth(180);
    m_sidebarPanel->setMaximumWidth(520);
    m_sidebarPanel->setStyleSheet("background:#0d0d0d;");

    QVBoxLayout* slay = new QVBoxLayout(m_sidebarPanel);
    slay->setContentsMargins(0, 0, 0, 0);
    slay->setSpacing(0);

    // ── Panel Title ────────────────────────────────────────────────
    QWidget* titleBar = new QWidget(m_sidebarPanel);
    titleBar->setFixedHeight(35);
    titleBar->setStyleSheet("background:#0d0d0d; border-bottom:1px solid #111111;");
    QHBoxLayout* tlay = new QHBoxLayout(titleBar);
    tlay->setContentsMargins(14, 0, 6, 0);
    QLabel* panelTitle = new QLabel("EXPLORER", titleBar);
    panelTitle->setStyleSheet("color:#888888; font-size:10px; font-weight:700; letter-spacing:1.5px;");
    tlay->addWidget(panelTitle, 1);

    // Action buttons in sidebar title
    auto mkTitleBtn = [&](const QString& icon, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(icon, titleBar);
        b->setFixedSize(22, 22);
        b->setToolTip(tip);
        b->setStyleSheet(
            "QPushButton{color:#555;font-size:14px;background:transparent;border:none;border-radius:3px;}"
            "QPushButton:hover{color:#ff6b35;background:rgba(255,107,53,0.08);}"
        );
        return b;
    };

    QPushButton* btnNewFile  = mkTitleBtn("📄", "New File");
    QPushButton* btnNewFolder= mkTitleBtn("📁", "New Folder");
    QPushButton* btnRefresh  = mkTitleBtn("↺", "Refresh Explorer");
    QPushButton* btnCollapse = mkTitleBtn("⇤", "Collapse All");
    QPushButton* btnAddOpen  = mkTitleBtn("＋", "Open Folder");

    connect(btnNewFile,  &QPushButton::clicked, this, &MokoWindow::newFile);
    connect(btnAddOpen,  &QPushButton::clicked, this, &MokoWindow::openFolder);
    connect(btnRefresh,  &QPushButton::clicked, [this]{
        if(m_fsModel) m_fsModel->setRootPath(m_workspaceRoot);
    });
    connect(btnCollapse, &QPushButton::clicked, [this]{
        if(m_fileTree) m_fileTree->collapseAll();
    });

    tlay->addWidget(btnNewFile);
    tlay->addWidget(btnNewFolder);
    tlay->addWidget(btnRefresh);
    tlay->addWidget(btnCollapse);
    tlay->addWidget(btnAddOpen);
    slay->addWidget(titleBar);

    // ── OPEN EDITORS Section ───────────────────────────────────────
    m_openEditorsSection = new QWidget(m_sidebarPanel);
    QVBoxLayout* oelay = new QVBoxLayout(m_openEditorsSection);
    oelay->setContentsMargins(0, 0, 0, 0);
    oelay->setSpacing(0);

    m_openEditorsList = new QListWidget(m_openEditorsSection);
    m_openEditorsList->setMaximumHeight(160);
    m_openEditorsList->setStyleSheet(
        "QListWidget { background:#0d0d0d; border:none; }"
        "QListWidget::item { padding:2px 10px 2px 24px; min-height:22px; font-size:12px; }"
        "QListWidget::item:hover { background:#141414; }"
        "QListWidget::item:selected { background:rgba(255,107,53,0.18); color:#ff6b35; }"
    );
    connect(m_openEditorsList, &QListWidget::itemClicked, [this](QListWidgetItem* item){
        QString path = item->data(Qt::UserRole).toString();
        if(!path.isEmpty()) openFile(path);
    });
    oelay->addWidget(m_openEditorsList);

    QWidget* oeHeader = makeSectionHeader("OPEN EDITORS", m_openEditorsSection);
    slay->addWidget(oeHeader);
    slay->addWidget(m_openEditorsSection);

    // ── Filter / Search ────────────────────────────────────────────
    m_searchBox = new QLineEdit(m_sidebarPanel);
    m_searchBox->setPlaceholderText("  🔍  Filter files…");
    m_searchBox->setStyleSheet(
        "QLineEdit { border:none; border-bottom:1px solid #111111; border-radius:0;"
        "background:#0d0d0d; color:#999; padding:5px 12px; font-size:12px; }"
        "QLineEdit:focus { border-bottom:1px solid #ff6b35; }"
    );
    connect(m_searchBox, &QLineEdit::textChanged, this, [this](const QString& text) {
        if (!m_fsModel) return;
        if (text.isEmpty())
            m_fsModel->setNameFilters({"*"});
        else
            m_fsModel->setNameFilters({"*" + text + "*"});
    });
    slay->addWidget(m_searchBox);

    // ── EXPLORER / File Tree Section ───────────────────────────────
    m_explorerSection = new QWidget(m_sidebarPanel);
    QVBoxLayout* exlay = new QVBoxLayout(m_explorerSection);
    exlay->setContentsMargins(0, 0, 0, 0);
    exlay->setSpacing(0);

    m_fileTree = new QTreeView(m_explorerSection);
    m_fileTree->header()->hide();
    m_fileTree->setAnimated(true);
    m_fileTree->setIndentation(16);
    m_fileTree->setUniformRowHeights(true);
    m_fileTree->setIconSize({16, 16});
    m_fileTree->setExpandsOnDoubleClick(false);   // We handle manually
    m_fileTree->setEditTriggers(QAbstractItemView::NoEditTriggers);
    connect(m_fileTree, &QTreeView::clicked,    this, &MokoWindow::onFileSingleClicked);
    connect(m_fileTree, &QTreeView::activated,  this, &MokoWindow::onFileActivated);
    exlay->addWidget(m_fileTree, 1);

    // Workspace name header
    QWidget* wsHeader = makeSectionHeader("WORKSPACE", m_explorerSection);
    slay->addWidget(wsHeader);
    slay->addWidget(m_explorerSection, 1);

    m_mainSplitter->addWidget(m_sidebarPanel);
}

// ─────────────────────────────────────────────────────────────────────────────
// Editor Area
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupEditorArea() {
    m_editorContainer = new QWidget(m_mainSplitter);
    QVBoxLayout* clay = new QVBoxLayout(m_editorContainer);
    clay->setContentsMargins(0, 0, 0, 0);
    clay->setSpacing(0);

    m_centerSplitter = new QSplitter(Qt::Vertical, m_editorContainer);
    m_centerSplitter->setHandleWidth(1);

    // ── Editor sub-container ───────────────────────────────────────
    QWidget* editorBox = new QWidget(m_centerSplitter);
    QVBoxLayout* elay  = new QVBoxLayout(editorBox);
    elay->setContentsMargins(0, 0, 0, 0);
    elay->setSpacing(0);

    // ── Editor Tab Widget ──────────────────────────────────────────
    m_editorTabs = new QTabWidget(editorBox);
    m_editorTabs->setObjectName("editorTabs");
    m_editorTabs->setTabsClosable(true);
    m_editorTabs->setMovable(true);
    m_editorTabs->setDocumentMode(true);
    m_editorTabs->setElideMode(Qt::ElideRight);

    // Style the tab bar with extra right-side corner buttons
    QWidget* cornerWidget = new QWidget();
    QHBoxLayout* cwlay = new QHBoxLayout(cornerWidget);
    cwlay->setContentsMargins(0, 0, 4, 0);
    cwlay->setSpacing(2);

    auto mkCornerBtn = [&](const QString& icon, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(icon);
        b->setFixedSize(24, 24);
        b->setToolTip(tip);
        b->setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#555;font-size:13px;border-radius:3px;}"
            "QPushButton:hover{background:rgba(255,107,53,0.08);color:#ff6b35;}"
        );
        return b;
    };

    QPushButton* splitBtn  = mkCornerBtn("⊟", "Split Editor");
    QPushButton* moreBtn   = mkCornerBtn("…", "More Actions");
    cwlay->addWidget(splitBtn);
    cwlay->addWidget(moreBtn);
    m_editorTabs->setCornerWidget(cornerWidget, Qt::TopRightCorner);

    connect(m_editorTabs, &QTabWidget::tabCloseRequested, this, &MokoWindow::onTabCloseRequested);

    QWidget* welcomeWidget = createWelcomeWidget();
    m_editorTabs->addTab(welcomeWidget, "Start");
    m_editorTabs->tabBar()->setTabButton(0, QTabBar::RightSide, nullptr);
    elay->addWidget(m_editorTabs, 1);

    // ── Breadcrumb ─────────────────────────────────────────────────
    m_breadcrumb = new QLabel("", editorBox);
    m_breadcrumb->setObjectName("breadcrumb");
    m_breadcrumb->setFixedHeight(25);
    m_breadcrumb->setTextFormat(Qt::RichText);
    elay->addWidget(m_breadcrumb);

    // ── Find Bar (hidden) ──────────────────────────────────────────
    m_findBar = new FindBar(editorBox);
    elay->addWidget(m_findBar);

    connect(m_editorTabs, &QTabWidget::currentChanged, this, &MokoWindow::onEditorTabChanged);

    m_centerSplitter->addWidget(editorBox);

    // ── Bottom Panel ───────────────────────────────────────────────
    setupBottomPanel();

    m_centerSplitter->setSizes({680, 270});
    clay->addWidget(m_centerSplitter, 1);
    m_mainSplitter->addWidget(m_editorContainer);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottom Panel
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupBottomPanel() {
    // Wrapper with top title bar (like VSCode panel)
    m_bottomPanel = new QWidget(m_centerSplitter);
    m_bottomPanel->setStyleSheet("background:#0d0d0d;");
    QVBoxLayout* bplay = new QVBoxLayout(m_bottomPanel);
    bplay->setContentsMargins(0, 0, 0, 0);
    bplay->setSpacing(0);

    // Panel top bar with tabs + controls
    m_bottomTabs = new QTabWidget(m_bottomPanel);
    m_bottomTabs->setObjectName("bottomTabs");
    m_bottomTabs->setTabPosition(QTabWidget::North);
    m_bottomTabs->setDocumentMode(true);

    // Corner controls for bottom panel
    QWidget* panelControls = new QWidget();
    QHBoxLayout* pclay = new QHBoxLayout(panelControls);
    pclay->setContentsMargins(0, 0, 4, 0);
    pclay->setSpacing(2);

    auto mkPanelBtn = [&](const QString& icon, const QString& tip) -> QPushButton* {
        QPushButton* b = new QPushButton(icon);
        b->setFixedSize(22, 22);
        b->setToolTip(tip);
        b->setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#555;font-size:12px;border-radius:3px;}"
            "QPushButton:hover{background:rgba(255,107,53,0.08);color:#ff6b35;}"
        );
        return b;
    };

    QPushButton* btnNewTerm  = mkPanelBtn("＋", "New Terminal");
    QPushButton* btnSplitTerm= mkPanelBtn("⊟", "Split Terminal");
    QPushButton* btnMaxPanel = mkPanelBtn("▲", "Maximize Panel");
    QPushButton* btnClosePanel= mkPanelBtn("✕", "Close Panel (Ctrl+`)");

    connect(btnNewTerm,   &QPushButton::clicked, this, &MokoWindow::addTerminalTab);
    connect(btnClosePanel,&QPushButton::clicked, this, &MokoWindow::toggleTerminal);
    connect(btnMaxPanel,  &QPushButton::clicked, [this]{
        m_centerSplitter->setSizes({0, 9999});
    });

    pclay->addWidget(btnNewTerm);
    pclay->addWidget(btnSplitTerm);
    pclay->addWidget(btnMaxPanel);
    pclay->addWidget(btnClosePanel);
    m_bottomTabs->setCornerWidget(panelControls, Qt::TopRightCorner);

    // TERMINAL tab
    m_terminal = new TerminalWidget();
    m_bottomTabs->addTab(m_terminal, "TERMINAL");

    // PROBLEMS tab
    m_problemsView = new QTextBrowser();
    m_problemsView->setStyleSheet(
        "background:#0a0a0a; color:#d0d0d0; border:none;"
        "font-family:'JetBrains Mono','Fira Code',monospace; font-size:12px;"
    );
    m_problemsView->setOpenLinks(false);
    m_bottomTabs->addTab(m_problemsView, "PROBLEMS");

    // OUTPUT tab
    m_outputView = new QTextBrowser();
    m_outputView->setStyleSheet(
        "background:#0a0a0a; color:#b0b0b0; border:none;"
        "font-family:'JetBrains Mono','Fira Code',monospace; font-size:12px;"
    );
    m_outputView->setText("[Moko IDE] Output panel ready.\n");
    m_bottomTabs->addTab(m_outputView, "OUTPUT");

    // DEBUG CONSOLE tab
    m_debugView = new QTextBrowser();
    m_debugView->setStyleSheet(
        "background:#0a0a0a; color:#b0b0b0; border:none;"
        "font-family:'JetBrains Mono','Fira Code',monospace; font-size:12px;"
    );
    m_debugView->setText("[Moko Debug] Console ready.\n");
    m_bottomTabs->addTab(m_debugView, "DEBUG CONSOLE");

    connect(m_bottomTabs, &QTabWidget::currentChanged, this, &MokoWindow::onBottomTabChanged);
    bplay->addWidget(m_bottomTabs, 1);

    m_centerSplitter->addWidget(m_bottomPanel);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat Panel
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupChatPanel() {
    m_chat = new ChatWidget(m_mainSplitter);
    m_chat->setMinimumWidth(300);
    m_chat->setMaximumWidth(560);
    m_mainSplitter->addWidget(m_chat);

    connect(m_chat, &ChatWidget::insertCodeToEditor, this, &MokoWindow::onInsertCodeToEditor);
    connect(m_chat, &ChatWidget::codeEditorContentRequested, this, [this]() {
        CodeEditor* ed = currentEditor();
        if (ed) {
            QString path = ed->property("filePath").toString();
            m_chat->receiveEditorSnapshot(ed->toPlainText(), m_lblLang->text().trimmed(), path);
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Status Bar
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setupStatusBar() {
    QStatusBar* sb = statusBar();
    sb->setFixedHeight(22);
    sb->setSizeGripEnabled(false);
    sb->setStyleSheet(
        "QStatusBar { background:#0d0d0d; border:none; border-top:1px solid #1a1a1a; padding:0; }"
        "QStatusBar QLabel { color:#888; padding:0 6px; font-size:11px; background:transparent; }"
        "QStatusBar QLabel:hover { background:rgba(255,107,53,0.10); color:#ff6b35; }"
        "QStatusBar::item { border:none; }"
    );

    // ── Left side ──────────────────────────────────────────────────
    m_lblBranch = new QLabel("  ⎇  moko-dev  ");
    m_lblBranch->setToolTip("Git Branch");
    m_lblBranch->setCursor(Qt::PointingHandCursor);
    sb->addWidget(m_lblBranch);

    m_lblHelper = new QLabel("  ✓ Clean  ");
    m_lblHelper->setToolTip("No issues detected");
    m_lblHelper->setCursor(Qt::PointingHandCursor);
    sb->addWidget(m_lblHelper);

    sb->addWidget(mkSep(sb));

    m_lblAiStatus = new QLabel("  🤖 MOKO  ");
    m_lblAiStatus->setToolTip("MOKO AI Status");
    sb->addWidget(m_lblAiStatus);

    // ── Right side (permanent) ─────────────────────────────────────
    m_lblSaved = new QLabel("  ");
    sb->addPermanentWidget(m_lblSaved);

    sb->addPermanentWidget(mkSep(sb));

    m_lblLinCol = new QLabel("  Ln 1, Col 1  ");
    m_lblLinCol->setToolTip("Line and Column");
    m_lblLinCol->setCursor(Qt::PointingHandCursor);
    m_lblLinCol->setMinimumWidth(90);
    m_lblLinCol->setAlignment(Qt::AlignCenter);
    sb->addPermanentWidget(m_lblLinCol);

    sb->addPermanentWidget(mkSep(sb));

    m_lblIndent = new QLabel("  Spaces: 4  ");
    m_lblIndent->setToolTip("Indentation");
    m_lblIndent->setCursor(Qt::PointingHandCursor);
    sb->addPermanentWidget(m_lblIndent);

    sb->addPermanentWidget(mkSep(sb));

    m_lblEncoding = new QLabel("  UTF-8  ");
    m_lblEncoding->setToolTip("File Encoding");
    m_lblEncoding->setCursor(Qt::PointingHandCursor);
    sb->addPermanentWidget(m_lblEncoding);

    sb->addPermanentWidget(mkSep(sb));

    m_lblEol = new QLabel("  LF  ");
    m_lblEol->setToolTip("End of Line Sequence");
    m_lblEol->setCursor(Qt::PointingHandCursor);
    sb->addPermanentWidget(m_lblEol);

    sb->addPermanentWidget(mkSep(sb));

    m_lblLang = new QLabel("  Plain Text  ");
    m_lblLang->setToolTip("Select Language Mode");
    m_lblLang->setCursor(Qt::PointingHandCursor);
    m_lblLang->setMinimumWidth(80);
    m_lblLang->setAlignment(Qt::AlignCenter);
    sb->addPermanentWidget(m_lblLang);

    sb->addPermanentWidget(mkSep(sb));

    m_lblWordWrap = new QLabel("  ↵  ");
    m_lblWordWrap->setToolTip("Toggle Word Wrap (Alt+Z)");
    m_lblWordWrap->setCursor(Qt::PointingHandCursor);
    sb->addPermanentWidget(m_lblWordWrap);

    sb->addPermanentWidget(mkSep(sb));

    m_lblCpu = new QLabel("  CPU: --  ");
    m_lblCpu->setMinimumWidth(80);
    m_lblCpu->setAlignment(Qt::AlignCenter);
    sb->addPermanentWidget(m_lblCpu);

    m_lblRam = new QLabel("  RAM: --  ");
    m_lblRam->setMinimumWidth(80);
    m_lblRam->setAlignment(Qt::AlignCenter);
    sb->addPermanentWidget(m_lblRam);
}

QLabel* MokoWindow::mkSep(QWidget* parent) {
    QLabel* s = new QLabel("|", parent);
    s->setStyleSheet("color:rgba(255,255,255,0.25); padding:0; font-size:10px;");
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// File Operations
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::setCwd(const QString& path) {
    if (!QDir(path).exists()) return;
    m_workspaceRoot = path;

    if (!m_fsModel) {
        m_fsModel = new QFileSystemModel(this);
        m_fsModel->setFilter(QDir::AllDirs | QDir::Files | QDir::NoDotAndDotDot);
        m_fsModel->setNameFilterDisables(false);
        m_fileTree->setModel(m_fsModel);
        for (int c = 1; c < m_fsModel->columnCount(); ++c)
            m_fileTree->hideColumn(c);
    }
    m_fsModel->setRootPath(path);
    m_fileTree->setRootIndex(m_fsModel->index(path));

    QString dirName = QDir(path).dirName();
    setWindowTitle(QString("%1 — Moko IDE").arg(dirName));
    if (m_terminal) m_terminal->setCwd(path);
    m_outputView->append(QString("<span style='color:#ff6b35'>[Moko IDE]</span> Workspace: <span style='color:#d0a060'>%1</span>").arg(path));
}

void MokoWindow::openFolder() {
    QString dir = QFileDialog::getExistingDirectory(
        this, "Open Folder", m_workspaceRoot,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) setCwd(dir);
}

void MokoWindow::newFile() {
    // Create a temporary unsaved file
    CodeEditor* editor = new CodeEditor(m_editorTabs);
    editor->setPlainText("");
    editor->setProperty("filePath", "");
    editor->setModified(false);
    editor->setMinimapVisible(m_minimapVisible);

    connect(editor, &CodeEditor::modifiedStateChanged, this, &MokoWindow::onEditorModified);
    connect(editor, &CodeEditor::cursorPositionChanged, this, [this, editor] {
        if (editor != currentEditor()) return;
        QTextCursor cur = editor->textCursor();
        m_lblLinCol->setText(QString("  Ln %1, Col %2  ")
            .arg(cur.blockNumber() + 1)
            .arg(cur.columnNumber() + 1));
    });

    // Remove welcome tab
    if (m_editorTabs->count() == 1 && m_editorTabs->tabText(0) == "Start") {
        m_editorTabs->removeTab(0);
        m_openTabs.clear();
    }

    int idx = m_editorTabs->addTab(editor, "Untitled");
    m_editorTabs->setCurrentIndex(idx);
    m_lblLang->setText("  Plain Text  ");
    m_breadcrumb->clear();
}

void MokoWindow::onFileSingleClicked(const QModelIndex& index) {
    if (!m_fsModel) return;
    QString path = m_fsModel->filePath(index);
    QFileInfo fi(path);
    if (fi.isDir()) {
        // Toggle expand/collapse
        if (m_fileTree->isExpanded(index))
            m_fileTree->collapse(index);
        else
            m_fileTree->expand(index);
        return;
    }
    // Open as preview (italic tab)
    openFile(path, true);
}

void MokoWindow::onFileActivated(const QModelIndex& index) {
    if (!m_fsModel) return;
    QString path = m_fsModel->filePath(index);
    if (QFileInfo(path).isDir()) return;
    openFile(path, false);  // Double-click → permanent tab
}

void MokoWindow::openFile(const QString& path, bool preview) {
    // If already open, just switch to it
    if (m_openTabs.contains(path)) {
        int idx = m_openTabs[path];
        m_editorTabs->setCurrentIndex(idx);
        // If it was preview, make it permanent on second open
        if (idx == m_previewTabIndex && !preview) {
            m_previewTabIndex = -1;
            m_previewTabPath.clear();
            QString name = QFileInfo(path).fileName();
            m_editorTabs->setTabText(idx, name);
        }
        return;
    }

    // If there's already a preview tab and we're in preview mode, close it first
    if (preview && m_previewTabIndex >= 0 && m_previewTabPath != path) {
        int pIdx = m_previewTabIndex;
        QString pPath = m_previewTabPath;
        m_openTabs.remove(pPath);
        m_editorTabs->removeTab(pIdx);
        // Recalculate indices
        for (auto it = m_openTabs.begin(); it != m_openTabs.end(); ++it)
            if (it.value() > pIdx) it.value()--;
        m_previewTabIndex = -1;
        m_previewTabPath.clear();
    }

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QString content = QTextStream(&f).readAll();

    CodeEditor* editor = new CodeEditor(m_editorTabs);
    editor->setPlainText(content);
    editor->setProperty("filePath", path);
    editor->setModified(false);
    editor->setMinimapVisible(m_minimapVisible);
    if (m_wordWrap)
        editor->setLineWrapMode(QPlainTextEdit::WidgetWidth);

    QString lang   = langFromPath(path);
    QString hlLang = highlightLang(lang);
    if (!hlLang.isEmpty())
        new SyntaxHighlighter(editor->document(), hlLang);

    connect(editor, &CodeEditor::modifiedStateChanged, this, &MokoWindow::onEditorModified);
    connect(editor, &CodeEditor::cursorPositionChanged, this, [this, editor] {
        if (editor != currentEditor()) return;
        QTextCursor cur = editor->textCursor();
        m_lblLinCol->setText(QString("  Ln %1, Col %2  ")
            .arg(cur.blockNumber() + 1)
            .arg(cur.columnNumber() + 1));
    });
    connect(editor, &QPlainTextEdit::textChanged, this, [this] {
        QTimer::singleShot(400, this, [this]{ runHelper(); });
    });

    // On first edit in preview mode → promote to permanent
    connect(editor, &QPlainTextEdit::textChanged, this, [this, path]() {
        if (m_previewTabPath == path) {
            m_previewTabIndex = -1;
            m_previewTabPath.clear();
            if (m_openTabs.contains(path)) {
                int idx = m_openTabs[path];
                QString name = QFileInfo(path).fileName();
                m_editorTabs->setTabText(idx, name);
            }
        }
    });

    // Remove welcome tab
    if (m_editorTabs->count() == 1 && m_editorTabs->tabText(0) == "Start") {
        m_editorTabs->removeTab(0);
        m_openTabs.clear();
    }

    QString ext  = QFileInfo(path).suffix().toLower();
    QString icon = iconForExtension(ext);
    QString name = QFileInfo(path).fileName();
    QString tabLabel = preview ? name : name;  // italic via stylesheet for preview

    int idx = m_editorTabs->addTab(editor, icon + " " + tabLabel);
    m_editorTabs->setCurrentIndex(idx);
    m_openTabs[path] = idx;

    if (preview) {
        m_previewTabIndex = idx;
        m_previewTabPath  = path;
        // Make tab text italic to indicate preview
        m_editorTabs->tabBar()->setTabData(idx, "preview");
    }

    m_lblLang->setText(QString("  %1  ").arg(lang));
    updateBreadcrumb(path);
    updateOpenEditorsList();
}

void MokoWindow::updateOpenEditorsList() {
    if (!m_openEditorsList) return;
    m_openEditorsList->clear();
    for (int i = 0; i < m_editorTabs->count(); ++i) {
        QWidget* w = m_editorTabs->widget(i);
        if (!w) continue;
        QString path = w->property("filePath").toString();
        if (path.isEmpty()) continue;
        QString ext  = QFileInfo(path).suffix().toLower();
        QString name = iconForExtension(ext) + "  " + QFileInfo(path).fileName();
        QListWidgetItem* item = new QListWidgetItem(name, m_openEditorsList);
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
    }
    // Adjust max height based on content
    int h = qMin(160, m_openEditorsList->count() * 24 + 4);
    m_openEditorsList->setMaximumHeight(h > 0 ? h : 0);
}

void MokoWindow::addTerminalTab() {
    m_terminalCount++;
    TerminalWidget* newTerm = new TerminalWidget();
    newTerm->setCwd(m_workspaceRoot);
    m_bottomTabs->addTab(newTerm, QString("TERMINAL %1").arg(m_terminalCount));
    m_bottomTabs->setCurrentWidget(newTerm);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab Management
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::onEditorTabChanged(int index) {
    if (index < 0) return;
    if (!m_breadcrumb || !m_lblLang) return;
    QWidget* w = m_editorTabs->widget(index);
    if (!w) return;
    QString path = w->property("filePath").toString();
    if (path.isEmpty()) {
        m_breadcrumb->clear();
        m_lblLang->setText("  Plain Text  ");
        return;
    }
    updateBreadcrumb(path);
    m_lblLang->setText(QString("  %1  ").arg(langFromPath(path)));
    if (auto* ed = currentEditor()) {
        QTextCursor cur = ed->textCursor();
        m_lblLinCol->setText(QString("  Ln %1, Col %2  ")
            .arg(cur.blockNumber()+1).arg(cur.columnNumber()+1));
        m_findBar->attachEditor(ed);
    }
    runHelper();
}

void MokoWindow::onTabCloseRequested(int index) {
    QWidget* w = m_editorTabs->widget(index);
    if (w) {
        QString path = w->property("filePath").toString();
        m_openTabs.remove(path);
        if (m_previewTabIndex == index) {
            m_previewTabIndex = -1;
            m_previewTabPath.clear();
        }
        for (auto it = m_openTabs.begin(); it != m_openTabs.end(); ++it)
            if (it.value() > index) it.value()--;
        if (m_previewTabIndex > index) m_previewTabIndex--;
    }
    m_editorTabs->removeTab(index);

    if (m_editorTabs->count() == 0) {
        m_breadcrumb->clear();
        m_lblLang->setText("  Plain Text  ");
        m_lblHelper->setText("  ✓ Clean  ");
        m_problemsView->clear();

        // Re-add welcome screen
        QWidget* welcomeWidget = createWelcomeWidget();
        m_editorTabs->addTab(welcomeWidget, "Start");
        m_editorTabs->tabBar()->setTabButton(0, QTabBar::RightSide, nullptr);
    }
    updateOpenEditorsList();
}

void MokoWindow::closeCurrentTab() {
    int idx = m_editorTabs->currentIndex();
    if (idx >= 0) onTabCloseRequested(idx);
}

void MokoWindow::onEditorModified(bool modified) {
    CodeEditor* ed = qobject_cast<CodeEditor*>(sender());
    if (!ed) return;
    for (int i = 0; i < m_editorTabs->count(); ++i) {
        if (m_editorTabs->widget(i) == ed) {
            QString path = ed->property("filePath").toString();
            QString ext  = path.isEmpty() ? "" : QFileInfo(path).suffix().toLower();
            QString icon = path.isEmpty() ? "📄" : iconForExtension(ext);
            QString name = path.isEmpty() ? "Untitled" : QFileInfo(path).fileName();
            // Modified: show dot ● before name
            m_editorTabs->setTabText(i, modified
                ? "● " + icon + " " + name
                : icon + " " + name);
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Save
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::saveCurrentFile() {
    CodeEditor* ed = currentEditor();
    if (!ed) return;
    QString path = ed->property("filePath").toString();

    if (path.isEmpty()) {
        // "Save As" for new files
        path = QFileDialog::getSaveFileName(this, "Save File", m_workspaceRoot);
        if (path.isEmpty()) return;
        ed->setProperty("filePath", path);
        m_openTabs[path] = m_editorTabs->currentIndex();
    }

    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream(&f) << ed->toPlainText();
        ed->setModified(false);
        m_lblSaved->setText("  ✓ Saved  ");
        m_lblSaved->setStyleSheet("color:#89d185;");
        QTimer::singleShot(2500, this, [this]{
            m_lblSaved->setText("  ");
            m_lblSaved->setStyleSheet("color:#fff;");
        });
        m_outputView->append(QString("<span style='color:#00ff88'>[Moko IDE]</span> Saved: <span style='color:#d0a060'>%1</span>").arg(path));
        updateOpenEditorsList();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
CodeEditor* MokoWindow::currentEditor() {
    return qobject_cast<CodeEditor*>(m_editorTabs->currentWidget());
}

void MokoWindow::updateBreadcrumb(const QString& path) {
    QString rel = QDir(m_workspaceRoot).relativeFilePath(path);
    QStringList parts = rel.split('/');
    QStringList formatted;
    for (int i = 0; i < parts.size(); ++i) {
        if (i == parts.size() - 1)
            formatted << QString("<span style='color:#cccccc;font-weight:500'>%1</span>").arg(parts[i].toHtmlEscaped());
        else
            formatted << QString("<span style='color:#555555'>%1</span>").arg(parts[i].toHtmlEscaped());
    }
    m_breadcrumb->setText("&nbsp;&nbsp;" + formatted.join(
        "&nbsp;<span style='color:#444444'> &gt; </span>&nbsp;"));
}

void MokoWindow::runHelper() {
    CodeEditor* ed = currentEditor();
    if (!ed) return;
    QString code = ed->toPlainText();
    QString path = ed->property("filePath").toString();
    m_lastIssues = m_helper.analyzeCode(code);

    int errCnt = 0, wrnCnt = 0;
    for (auto& iss : m_lastIssues) {
        if (iss.severity == "ERROR") errCnt++;
        else wrnCnt++;
    }

    if (errCnt > 0) {
        m_lblHelper->setText(QString("  ✗ %1 error%2  ").arg(errCnt).arg(errCnt > 1 ? "s" : ""));
        m_lblHelper->setStyleSheet("color:#f48771; font-weight:600;");
    } else if (wrnCnt > 0) {
        m_lblHelper->setText(QString("  ⚠ %1 warning%2  ").arg(wrnCnt).arg(wrnCnt > 1 ? "s" : ""));
        m_lblHelper->setStyleSheet("color:#cca700; font-weight:600;");
    } else {
        m_lblHelper->setText("  ✓ Clean  ");
        m_lblHelper->setStyleSheet("color:#89d185;");
    }

    // Update PROBLEMS tab
    m_problemsView->clear();
    QString fname = path.isEmpty() ? "Untitled" : QFileInfo(path).fileName();
    if (m_lastIssues.isEmpty()) {
        m_problemsView->setHtml("<span style='color:#555'>No problems detected in workspace.</span>");
        m_bottomTabs->setTabText(1, "PROBLEMS");
    } else {
        m_bottomTabs->setTabText(1, QString("PROBLEMS (%1)").arg(m_lastIssues.size()));
        QString html;
        for (auto& iss : m_lastIssues) {
            QString icon  = (iss.severity == "ERROR") ? "✗" : "⚠";
            QString color = (iss.severity == "ERROR") ? "#f48771" : "#cca700";
            html += QString("<div style='padding:2px 0'>"
                           "<span style='color:%3; font-weight:bold'>%1</span>&nbsp;&nbsp;"
                           "<span style='color:#cccccc'>%2</span>&nbsp;&nbsp;"
                           "<span style='color:#555555; font-size:11px'>%4&nbsp;Ln %5</span>"
                           "</div>")
                   .arg(icon).arg(iss.message.toHtmlEscaped()).arg(color)
                   .arg(fname).arg(iss.line);
        }
        m_problemsView->setHtml(html);
    }

    // Feed context to AI chat
    QString ctx = m_helper.generateCompressedContext(code, fname);
    if (m_chat) m_chat->setCodeContext(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Toggles
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::toggleSidebar() {
    bool vis = !m_sidebarPanel->isVisible();
    m_sidebarPanel->setVisible(vis);
    m_btnExplorer->setChecked(vis);
}

void MokoWindow::toggleChat() {
    bool vis = !m_chat->isVisible();
    m_chat->setVisible(vis);
    m_btnChat->setChecked(vis);
}

void MokoWindow::toggleTerminal() {
    m_terminalVisible = !m_terminalVisible;
    m_bottomPanel->setVisible(m_terminalVisible);
    if (m_terminalVisible) {
        m_centerSplitter->setSizes({660, 270});
        if (m_terminal) m_terminal->setFocus();
    }
}

void MokoWindow::toggleWordWrap() {
    m_wordWrap = !m_wordWrap;
    // Apply to all open editors
    for (int i = 0; i < m_editorTabs->count(); ++i) {
        if (auto* ed = qobject_cast<CodeEditor*>(m_editorTabs->widget(i))) {
            ed->setLineWrapMode(m_wordWrap
                ? QPlainTextEdit::WidgetWidth
                : QPlainTextEdit::NoWrap);
        }
    }
    m_lblWordWrap->setStyleSheet(m_wordWrap ? "color:#ff6b35;" : "color:#888;");
}

void MokoWindow::toggleMinimap() {
    m_minimapVisible = !m_minimapVisible;
    for (int i = 0; i < m_editorTabs->count(); ++i) {
        if (auto* ed = qobject_cast<CodeEditor*>(m_editorTabs->widget(i))) {
            ed->setMinimapVisible(m_minimapVisible);
        }
    }
}

void MokoWindow::showGitPanel() {
    m_btnGit->setChecked(!m_btnGit->isChecked());
    // Placeholder — show info in output
    m_outputView->append("<span style='color:#ff6b35'>[Git]</span> Source Control panel — coming soon.");
    m_bottomTabs->setCurrentWidget(m_outputView);
    if (!m_terminalVisible) toggleTerminal();
}

void MokoWindow::onBottomTabChanged(int index) {
    if (index == 0 && m_terminal) m_terminal->setFocus();
}

// ─────────────────────────────────────────────────────────────────────────────
// Find Bar
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::showFindBar() {
    if (auto* ed = currentEditor()) {
        m_findBar->attachEditor(ed);
        m_findBar->show();
        m_findBar->focusFind();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Command Palette  (VSCode-style)
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::showCommandPalette() {
    QDialog* dlg = new QDialog(this, Qt::Popup | Qt::FramelessWindowHint);
    dlg->setFixedSize(640, 400);
    dlg->setStyleSheet(R"(
        QDialog {
            background: #0d0d0d;
            border: 1px solid #222222;
            border-radius: 8px;
        }
        QLineEdit {
            background: #0a0a0a;
            color: #ffffff;
            border: none;
            border-radius: 0;
            border-bottom: 2px solid #ff6b35;
            padding: 10px 14px;
            font-size: 14px;
            selection-background-color: rgba(255, 107, 53, 0.25);
        }
        QListWidget {
            background: #0d0d0d;
            color: #b0b0b0;
            border: none;
            font-size: 12px;
            outline: none;
        }
        QListWidget::item {
            padding: 7px 14px;
            border-bottom: 1px solid #141414;
        }
        QListWidget::item:selected {
            background: rgba(255, 107, 53, 0.18);
            color: #ff6b35;
        }
        QListWidget::item:hover {
            background: #141414;
        }
    )");

    QVBoxLayout* lay = new QVBoxLayout(dlg);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(0);

    QLineEdit* input = new QLineEdit(dlg);
    input->setPlaceholderText("  > Type a command or search file…");
    lay->addWidget(input);

    // Category separator label
    QLabel* catLabel = new QLabel("  Recently Used", dlg);
    catLabel->setStyleSheet(
        "background:#0a0a0a; color:#555555; font-size:10px; font-weight:700; "
        "letter-spacing:1px; padding:4px 14px; border-bottom:1px solid #141414;"
    );
    lay->addWidget(catLabel);

    QListWidget* list = new QListWidget(dlg);
    lay->addWidget(list, 1);

    // Status hint bar
    QLabel* hintBar = new QLabel("  ↑↓ to navigate   Enter to select   Esc to dismiss", dlg);
    hintBar->setStyleSheet(
        "background:#0a0a0a; color:#555555; font-size:10px; padding:4px 14px;"
        "border-top:1px solid #141414;"
    );
    lay->addWidget(hintBar);

    struct Cmd {
        QString icon;
        QString label;
        QString shortcut;
        QString category;
        std::function<void()> fn;
    };

    QVector<Cmd> commands = {
        {"📂", "File: Open Folder",              "Ctrl+K Ctrl+O", "File",   [this]{ openFolder(); }},
        {"📄", "File: New File",                  "Ctrl+N",        "File",   [this]{ newFile(); }},
        {"💾", "File: Save",                      "Ctrl+S",        "File",   [this]{ saveCurrentFile(); }},
        {"❌", "File: Close Tab",                 "Ctrl+W",        "File",   [this]{ closeCurrentTab(); }},
        {"🔍", "Edit: Find / Replace",            "Ctrl+F",        "Edit",   [this]{ showFindBar(); }},
        {"↩", "Edit: Toggle Word Wrap",           "Alt+Z",         "Edit",   [this]{ toggleWordWrap(); }},
        {"📁", "View: Toggle Explorer",           "Ctrl+Shift+E",  "View",   [this]{ toggleSidebar(); }},
        {"💻", "View: Toggle Terminal",           "Ctrl+`",        "View",   [this]{ toggleTerminal(); }},
        {"🤖", "View: Toggle AI Chat",            "Ctrl+Shift+G",  "View",   [this]{ toggleChat(); }},
        {"🗺", "View: Toggle Minimap",            "",              "View",   [this]{ toggleMinimap(); }},
        {"⎇",  "View: Source Control (Git)",      "",              "View",   [this]{ showGitPanel(); }},
        {"🔧", "Moko: Run Jantung Helper",        "",              "Moko",   [this]{ runHelper(); }},
        {"⚙",  "Preferences: Open Settings",     "Ctrl+,",        "Moko",   [this]{ openSettings(); }},
        {"🚪", "Application: Quit",               "Ctrl+Q",        "App",    []{ qApp->quit(); }},
    };

    auto populate = [&](const QString& filter) {
        list->clear();
        bool isCmd = filter.startsWith(">");
        QString q = isCmd ? filter.mid(1).trimmed() : filter.trimmed();

        catLabel->setText(q.isEmpty() ? "  Recently Used" : QString("  Results (%1)").arg("filtered"));

        for (int i = 0; i < commands.size(); ++i) {
            bool match = q.isEmpty()
                || commands[i].label.toLower().contains(q.toLower())
                || commands[i].shortcut.toLower().contains(q.toLower());
            if (!match) continue;

            // Custom item widget
            QWidget* itemWidget = new QWidget();
            itemWidget->setStyleSheet("background:transparent;");
            QHBoxLayout* ilay = new QHBoxLayout(itemWidget);
            ilay->setContentsMargins(14, 4, 14, 4);
            ilay->setSpacing(10);

            QLabel* iconLbl = new QLabel(commands[i].icon);
            iconLbl->setStyleSheet("font-size:14px; color:#cccccc; min-width:20px;");
            iconLbl->setFixedWidth(22);

            QLabel* lblText = new QLabel(commands[i].label);
            lblText->setStyleSheet("font-size:12px; color:#cccccc;");

            QLabel* lblShortcut = new QLabel(commands[i].shortcut);
            lblShortcut->setStyleSheet("font-size:11px; color:#555; text-align:right;");
            lblShortcut->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

            ilay->addWidget(iconLbl);
            ilay->addWidget(lblText, 1);
            ilay->addWidget(lblShortcut);

            QListWidgetItem* item = new QListWidgetItem(list);
            item->setData(Qt::UserRole, i);
            item->setSizeHint({0, 34});
            list->addItem(item);
            list->setItemWidget(item, itemWidget);
        }
        if (list->count() > 0) list->setCurrentRow(0);
    };
    populate(">");

    auto execCmd = [&]{
        QListWidgetItem* item = list->currentItem();
        if (!item) return;
        int idx = item->data(Qt::UserRole).toInt();
        dlg->accept();
        if (idx >= 0 && idx < commands.size())
            commands[idx].fn();
    };

    connect(input, &QLineEdit::textChanged, [&](const QString& t){ populate(t.isEmpty() ? ">" : t); });
    connect(list, &QListWidget::itemActivated, [&](QListWidgetItem*){ execCmd(); });
    connect(input, &QLineEdit::returnPressed, [&]{ execCmd(); });

    // Arrow key navigation
    input->installEventFilter(new QObject(dlg));

    // Position below menu bar, centered
    QPoint pos = mapToGlobal(QPoint(geometry().width()/2 - 320, 30));
    dlg->move(pos);
    input->setFocus();
    dlg->exec();
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware Stats
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::updateHardwareStats() {
    QFile la("/proc/loadavg");
    if (la.open(QIODevice::ReadOnly)) {
        double load = la.readLine().trimmed().split(' ')[0].toDouble();
        int pct = qMin(100, (int)(load * 10));
        QString color = pct > 80 ? "#f48771" : (pct > 50 ? "#cca700" : "#ffffff");
        m_lblCpu->setText(QString("  CPU %1%  ").arg(pct));
        m_lblCpu->setStyleSheet(QString("color:%1;").arg(color));
    }

    QFile mi("/proc/meminfo");
    if (mi.open(QIODevice::ReadOnly)) {
        long long total = 0, avail = 0;
        while (!mi.atEnd()) {
            QString l = mi.readLine();
            if (l.startsWith("MemTotal:"))     total = l.split(':')[1].trimmed().split(' ')[0].toLongLong();
            if (l.startsWith("MemAvailable:")) avail = l.split(':')[1].trimmed().split(' ')[0].toLongLong();
        }
        if (total > 0) {
            int ramPct = (int)((total - avail) * 100 / total);
            QString color = ramPct > 85 ? "#f48771" : (ramPct > 70 ? "#cca700" : "#ffffff");
            m_lblRam->setText(QString("  RAM %1%  ").arg(ramPct));
            m_lblRam->setStyleSheet(QString("color:%1;").arg(color));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::openSettings() {
    SettingsDialog dlg(this);
    dlg.exec();
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert Code to Editor (from AI Chat)
// ─────────────────────────────────────────────────────────────────────────────
void MokoWindow::onInsertCodeToEditor(const QString& code, const QString& lang) {
    CodeEditor* ed = currentEditor();
    if (!ed) {
        QString ext = "txt";
        QString lowerLang = lang.toLower().trimmed();
        if (lowerLang == "python")                        ext = "py";
        else if (lowerLang == "c++" || lowerLang == "cpp") ext = "cpp";
        else if (lowerLang == "javascript" || lowerLang == "js") ext = "js";
        else if (lowerLang == "typescript" || lowerLang == "ts") ext = "ts";
        else if (lowerLang == "html")                     ext = "html";
        else if (lowerLang == "css")                      ext = "css";
        else if (lowerLang == "rust" || lowerLang == "rs")ext = "rs";
        else if (lowerLang == "shell" || lowerLang == "sh")ext = "sh";

        QString tempPath = m_workspaceRoot + "/moko_generated." + ext;
        QFile file(tempPath);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << code;
            file.close();
        }
        openFile(tempPath);
        return;
    }
    QTextCursor cursor = ed->textCursor();
    cursor.insertText(code);
    ed->setModified(true);
}

QWidget* MokoWindow::createWelcomeWidget() {
    QScrollArea* scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setStyleSheet("border:none; background:#0a0a0a;");

    QWidget* welcome = new QWidget();
    welcome->setStyleSheet("background:#0a0a0a;");
    QVBoxLayout* wlay = new QVBoxLayout(welcome);
    wlay->setAlignment(Qt::AlignCenter);
    wlay->setSpacing(14);

    QLabel* logo = new QLabel("MOKO");
    logo->setAlignment(Qt::AlignCenter);
    logo->setStyleSheet(
        "font-size: 52px; font-weight: 900; color: #ff6b35; "
        "letter-spacing: 12px; font-family: 'Inter','Segoe UI',sans-serif; "
        "border: 2px solid rgba(255, 107, 53, 0.35); border-radius: 12px; "
        "padding: 10px 36px; background: rgba(255, 107, 53, 0.04);"
    );

    QLabel* wTitle = new QLabel("Moko IDE");
    wTitle->setAlignment(Qt::AlignCenter);
    wTitle->setStyleSheet("font-size: 32px; font-weight: 800; color: #ffffff; letter-spacing: 1px;");

    QLabel* wSubtitle = new QLabel("Native C++ · Qt5 · AI-Powered Development");
    wSubtitle->setAlignment(Qt::AlignCenter);
    wSubtitle->setStyleSheet("font-size: 13px; color: #666666;");

    // Quick action grid
    QWidget* grid = new QWidget();
    QHBoxLayout* gridlay = new QHBoxLayout(grid);
    gridlay->setSpacing(12);

    auto mkQuickBtn = [&](const QString& icon, const QString& label, const QString& sub, std::function<void()> fn) -> QWidget* {
        QFrame* card = new QFrame();
        card->setObjectName("welcomeCard");
        card->setFrameShape(QFrame::NoFrame);
        card->setStyleSheet(
            "QFrame#welcomeCard { background:#0d0d0d; border:1px solid #222222; border-radius:8px; padding:12px; }"
            "QFrame#welcomeCard:hover { background:#141414; border-color:#ff6b35; }"
        );
        card->setCursor(Qt::PointingHandCursor);
        card->setFixedSize(180, 90);
        QVBoxLayout* cl = new QVBoxLayout(card);
        cl->setContentsMargins(12, 8, 12, 8);
        cl->setSpacing(4);
        QLabel* ic = new QLabel(icon + "  " + label);
        ic->setStyleSheet("font-size:13px; font-weight:600; color:#cccccc; border:none; background:transparent;");
        QLabel* sc = new QLabel(sub);
        sc->setStyleSheet("font-size:11px; color:#555555; border:none; background:transparent;");
        cl->addWidget(ic);
        cl->addWidget(sc);

        // Make clickable overlay button
        QPushButton* overlay = new QPushButton("", card);
        overlay->setGeometry(0, 0, 180, 90);
        overlay->setStyleSheet("background:transparent; border:none;");
        connect(overlay, &QPushButton::clicked, fn);
        return card;
    };

    gridlay->addWidget(mkQuickBtn("📂", "Open Folder", "Ctrl+K Ctrl+O", [this]{ openFolder(); }));
    gridlay->addWidget(mkQuickBtn("📄", "New File", "Ctrl+N", [this]{ newFile(); }));
    gridlay->addWidget(mkQuickBtn("⌘", "Command Palette", "Ctrl+Shift+P", [this]{ showCommandPalette(); }));
    gridlay->addWidget(mkQuickBtn("⚙", "Settings", "Ctrl+,", [this]{ openSettings(); }));

    wlay->addSpacing(30);
    wlay->addWidget(logo, 0, Qt::AlignCenter);
    wlay->addSpacing(10);
    wlay->addWidget(wTitle);
    wlay->addWidget(wSubtitle);
    wlay->addSpacing(25);
    wlay->addWidget(grid, 0, Qt::AlignCenter);
    wlay->addSpacing(15);
    QLabel* hint = new QLabel("Press <b style='color:#ff6b35'>Ctrl+Shift+P</b> to open Command Palette");
    hint->setAlignment(Qt::AlignCenter);
    hint->setStyleSheet("color:#444444; font-size:12px;");
    hint->setTextFormat(Qt::RichText);
    wlay->addWidget(hint);

    scrollArea->setWidget(welcome);
    return scrollArea;
}

