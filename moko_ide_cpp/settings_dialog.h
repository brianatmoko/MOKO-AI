#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QFile>
#include <QDir>
#include <QHeaderView>
#include <QMessageBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QCheckBox>
#include <QTimer>

class SettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit SettingsDialog(QWidget* parent = nullptr);
    ~SettingsDialog();

private slots:
    void addRow();
    void removeSelectedRow();
    void saveAndClose();
    void onCellWidgetChanged();
    void onLocalLLMToggled(bool checked);
    void onModeChanged();
    void updateLocalLLMStatus();

private:
    void loadSettings();
    void applyDialogStyle();
    QString findConfigPath();
    QString findSettingsPath();
    void loadGlobalSettings();
    void saveGlobalSettings();

    QTableWidget* m_table;
    QPushButton*  m_btnAdd;
    QPushButton*  m_btnRemove;
    QPushButton*  m_btnSave;
    QPushButton*  m_btnCancel;

    // ── System Mode ────────────────────────────────────────────────────────
    QRadioButton*  m_radioAgent;
    QRadioButton*  m_radioRotation;
    QButtonGroup*  m_modeGroup;

    // ── Local LLM ─────────────────────────────────────────────────────────
    QCheckBox*  m_chkLocalLLM;
    QLabel*     m_lblLocalStatus;

    QString m_configPath;
    QString m_settingsPath;
};

#endif // SETTINGS_DIALOG_H
