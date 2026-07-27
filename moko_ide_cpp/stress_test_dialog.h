#ifndef STRESS_TEST_DIALOG_H
#define STRESS_TEST_DIALOG_H

#include <QDialog>
#include <QThread>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QJsonObject>

class QLabel;
class QPushButton;
class QTextBrowser;
class QProgressBar;
class QComboBox;
class QCheckBox;
class QFrame;

// ─── StressTestWorker (QThread) ───────────────────────────
class StressTestWorker : public QThread {
    Q_OBJECT
public:
    StressTestWorker(const QString& folderPath, const QString& testType, bool enableDeepCot, QObject* parent = nullptr);
    void cancel();

signals:
    void progressSignal(int percent, const QString& logText);
    void metricSignal(const QJsonObject& data);
    void finishedSignal(const QString& reportHeader);

protected:
    void run() override;

private:
    QString m_folderPath;
    QString m_testType;
    bool m_enableDeepCot;
    bool m_cancelled;

    void scanFiles(const QString& rootPath, QStringList& files, int& totalLoc);
    int estimateCyclomatic(const QStringList& files);
};

// ─── MokoStressTestDialog (QDialog) ───────────────────────
class MokoStressTestDialog : public QDialog {
    Q_OBJECT
public:
    explicit MokoStressTestDialog(QWidget* parent = nullptr);
    ~MokoStressTestDialog();

private slots:
    void browseFolder();
    void clearAll();
    void startTest();
    void stopTest();
    void onProgress(int pct, const QString& msg);
    void onMetrics(const QJsonObject& data);
    void onFinished(const QString& reportHeader);
    void onWorkerDone();
    void handleModelReply();

private:
    void buildUi();
    void runModelQuery(const QString& prompt, const QString& reportHeader);

    QString m_defaultWs;
    StressTestWorker* m_worker = nullptr;
    QNetworkAccessManager* m_networkManager;
    QNetworkReply* m_modelReply = nullptr;

    QLabel* m_lblWs;
    QComboBox* m_cbType;
    QCheckBox* m_chkCot;
    QTextBrowser* m_console;
    QProgressBar* m_prog;

    QLabel* m_mFiles;
    QLabel* m_mLoc;
    QLabel* m_mCc;
    QLabel* m_mVram;
    QLabel* m_mRam;
    QLabel* m_mComplex;

    QPushButton* m_btnBrowse;
    QPushButton* m_btnClear;
    QPushButton* m_btnStop;
    QPushButton* m_btnStart;

    QString m_accumulatedHeader;
};

#endif // STRESS_TEST_DIALOG_H
