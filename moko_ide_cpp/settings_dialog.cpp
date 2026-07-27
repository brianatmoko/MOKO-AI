#include "settings_dialog.h"
#include <QCheckBox>
#include <QDebug>
#include <QCoreApplication>

SettingsDialog::SettingsDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("Moko Settings — API & Worker Pool");
    resize(960, 680);
    applyDialogStyle();

    QVBoxLayout* mainLay = new QVBoxLayout(this);
    mainLay->setContentsMargins(16, 16, 16, 16);
    mainLay->setSpacing(12);

    QLabel* title = new QLabel("Configure Worker APIs & Foreman (Mandor)", this);
    title->setStyleSheet("font-size: 16px; font-weight: bold; color: #569cd6;");
    mainLay->addWidget(title);

    QLabel* desc = new QLabel(
        "Manage API endpoints, keys, and select who becomes the Foreman (Mandor).\n"
        "The Mandor will automatically coordinate tasks and manage code generation worker pools.", this);
    desc->setStyleSheet("color: #888; font-size: 11px;");
    mainLay->addWidget(desc);

    // ── System Mode Group ─────────────────────────────────────────────────────
    QGroupBox* modeBox = new QGroupBox("⚙  System Mode", this);
    modeBox->setStyleSheet(R"(
        QGroupBox {
            color: #569cd6;
            font-size: 12px;
            font-weight: bold;
            border: 1px solid rgba(86,156,214,0.35);
            border-radius: 6px;
            margin-top: 6px;
            padding: 10px 12px 10px 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
        }
    )");
    QHBoxLayout* modeLayout = new QHBoxLayout(modeBox);
    modeLayout->setSpacing(24);

    m_radioAgent = new QRadioButton("🤖  Agent AI Mode", this);
    m_radioAgent->setToolTip(
        "API bekerja sebagai tim kolaboratif:\n"
        "• Mandor (Foreman) merencanakan & menganalisis\n"
        "• Pekerja membangun secara paralel\n"
        "• Guard (Lokal) memverifikasi\n\n"
        "✗  API Rotation otomatis DINONAKTIFKAN"
    );
    m_radioAgent->setStyleSheet(
        "QRadioButton { color: #ffd700; font-size: 12px; font-weight: 600; }"
        "QRadioButton::indicator:checked { background: rgba(255,215,0,0.2); border: 2px solid #ffd700; border-radius: 6px; width: 12px; height: 12px; }"
        "QRadioButton::indicator:unchecked { border: 1px solid #555; border-radius: 6px; width: 12px; height: 12px; }"
    );

    m_radioRotation = new QRadioButton("🔄  API Rotation Mode", this);
    m_radioRotation->setToolTip(
        "API digunakan bergantian:\n"
        "• Jika API pertama kena rate-limit (HTTP 429) / timeout\n"
        "• Sistem otomatis beralih ke API berikutnya\n"
        "• Jika semua habis → fallback ke model lokal\n\n"
        "✗  Agent AI collaboration DINONAKTIFKAN"
    );
    m_radioRotation->setStyleSheet(
        "QRadioButton { color: #00e6ff; font-size: 12px; font-weight: 600; }"
        "QRadioButton::indicator:checked { background: rgba(0,230,255,0.2); border: 2px solid #00e6ff; border-radius: 6px; width: 12px; height: 12px; }"
        "QRadioButton::indicator:unchecked { border: 1px solid #555; border-radius: 6px; width: 12px; height: 12px; }"
    );

    m_modeGroup = new QButtonGroup(this);
    m_modeGroup->addButton(m_radioAgent,    0);
    m_modeGroup->addButton(m_radioRotation, 1);
    m_radioAgent->setChecked(true); // default

    // Description labels
    QLabel* lblAgent = new QLabel("<span style='color:#888; font-size:10px;'>Mandor + Pekerja + Guard bekerja sama. Cocok untuk membangun proyek besar.</span>", this);
    QLabel* lblRotation = new QLabel("<span style='color:#888; font-size:10px;'>Auto-ganti API jika limit. Cocok untuk request cepat & volume tinggi.</span>", this);
    lblAgent->setWordWrap(true);
    lblRotation->setWordWrap(true);

    QVBoxLayout* agentCol = new QVBoxLayout();
    agentCol->addWidget(m_radioAgent);
    agentCol->addWidget(lblAgent);

    QVBoxLayout* rotCol = new QVBoxLayout();
    rotCol->addWidget(m_radioRotation);
    rotCol->addWidget(lblRotation);

    modeLayout->addLayout(agentCol);
    modeLayout->addLayout(rotCol);
    modeLayout->addStretch();
    mainLay->addWidget(modeBox);

    // Wire mode radio buttons to status updater
    connect(m_radioAgent,    &QRadioButton::toggled, this, &SettingsDialog::onModeChanged);
    connect(m_radioRotation, &QRadioButton::toggled, this, &SettingsDialog::onModeChanged);

    // ── Local LLM Settings ────────────────────────────────────────────────────
    QGroupBox* localBox = new QGroupBox("🖥  Local LLM Settings", this);
    localBox->setStyleSheet(R"(
        QGroupBox {
            color: #00ff88;
            font-size: 12px;
            font-weight: bold;
            border: 1px solid rgba(0,255,136,0.3);
            border-radius: 6px;
            margin-top: 6px;
            padding: 10px 12px 10px 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
        }
    )");
    QVBoxLayout* localVLay = new QVBoxLayout(localBox);
    localVLay->setSpacing(8);

    // Toggle row
    QHBoxLayout* localToggleRow = new QHBoxLayout();

    m_chkLocalLLM = new QCheckBox("⚡  Aktifkan LLM Lokal (moko-coder-1.5b)", this);
    m_chkLocalLLM->setStyleSheet(
        "QCheckBox { color: #00ff88; font-size: 12px; font-weight: 600; }"
        "QCheckBox::indicator { width: 14px; height: 14px; }"
        "QCheckBox::indicator:unchecked { background: rgba(0,0,0,0.4); border: 1px solid rgba(0,255,136,0.3); border-radius: 3px; }"
        "QCheckBox::indicator:checked   { background: rgba(0,255,136,0.2); border: 2px solid #00ff88; border-radius: 3px; }"
    );
    connect(m_chkLocalLLM, &QCheckBox::toggled, this, &SettingsDialog::onLocalLLMToggled);

    localToggleRow->addWidget(m_chkLocalLLM);
    localToggleRow->addStretch();

    // Status badge
    m_lblLocalStatus = new QLabel("⭕  Status: Tidak Aktif", this);
    m_lblLocalStatus->setStyleSheet(
        "color: #555; font-size: 11px; font-weight: 600; "
        "background: rgba(0,0,0,0.3); border-radius: 4px; padding: 3px 10px;"
    );
    localToggleRow->addWidget(m_lblLocalStatus);

    localVLay->addLayout(localToggleRow);

    // Description that changes with mode
    QLabel* descLocal = new QLabel(
        "<span style='color:#668; font-size:10px;'>"
        "<b>API ON + Lokal ON</b> → LLM Lokal berstatus <b style='color:#ffd700;'>Belajar</b> "
        "(distilasi dari jawaban API guru, membangun SFT dataset otomatis).<br>"
        "<b>API OFF + Lokal ON</b> → LLM Lokal berstatus <b style='color:#00e6ff;'>Mandor + Eksekutor Otonom</b> "
        "(4-Phase: Mandor → Validator → Eksekutor → Validator)."
        "</span>", this);
    descLocal->setWordWrap(true);
    localVLay->addWidget(descLocal);

    mainLay->addWidget(localBox);

    // ── API Table Label ───────────────────────────────────────────────────────
    QLabel* apiLabel = new QLabel("API Worker Pool:", this);
    apiLabel->setStyleSheet("color: #888; font-size: 11px; margin-top: 4px;");
    mainLay->addWidget(apiLabel);


    m_table = new QTableWidget(0, 7, this);
    m_table->setHorizontalHeaderLabels({"Mandor?", "Active?", "Name", "Provider", "Model Name", "API Key(s)", "API Base / Endpoint"});
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::SingleSelection);
    m_table->verticalHeader()->setVisible(false);
    
    // Column widths
    m_table->setColumnWidth(0, 80);
    m_table->setColumnWidth(1, 80);
    m_table->setColumnWidth(2, 120);
    m_table->setColumnWidth(3, 100);
    m_table->setColumnWidth(4, 140);
    m_table->setColumnWidth(5, 180);

    m_table->setStyleSheet(R"(
        QTableWidget {
            background-color: #1e1e1e;
            gridline-color: #2d2d2d;
            border: 1px solid #3c3c3c;
            color: #d4d4d4;
            font-size: 12px;
        }
        QHeaderView::section {
            background-color: #2d2d2d;
            color: #cccccc;
            padding: 6px;
            border: 1px solid #1e1e1e;
            font-weight: bold;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #094771;
            color: #ffffff;
        }
    )");
    mainLay->addWidget(m_table, 1);

    // ── Buttons ──────────────────────────────────────────────────────────────
    QHBoxLayout* btnRow = new QHBoxLayout();
    
    m_btnAdd = new QPushButton("⊕ Add API", this);
    m_btnAdd->setStyleSheet("QPushButton { background-color: #0e639c; color: white; padding: 6px 12px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #1177bb; }");
    connect(m_btnAdd, &QPushButton::clicked, this, &SettingsDialog::addRow);
    btnRow->addWidget(m_btnAdd);

    m_btnRemove = new QPushButton("⊖ Remove Selected", this);
    m_btnRemove->setStyleSheet("QPushButton { background-color: #5a1d1d; color: white; padding: 6px 12px; border-radius: 3px; } QPushButton:hover { background-color: #7a2d2d; }");
    connect(m_btnRemove, &QPushButton::clicked, this, &SettingsDialog::removeSelectedRow);
    btnRow->addWidget(m_btnRemove);

    btnRow->addStretch(1);

    m_btnCancel = new QPushButton("Cancel", this);
    m_btnCancel->setStyleSheet("QPushButton { background-color: #3c3c3c; color: #ccc; padding: 6px 14px; border-radius: 3px; } QPushButton:hover { background-color: #4c4c4c; color: #fff; }");
    connect(m_btnCancel, &QPushButton::clicked, this, &QDialog::reject);
    btnRow->addWidget(m_btnCancel);

    m_btnSave = new QPushButton("Save Config", this);
    m_btnSave->setStyleSheet("QPushButton { background-color: #007acc; color: white; padding: 6px 16px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #0098ff; }");
    connect(m_btnSave, &QPushButton::clicked, this, &SettingsDialog::saveAndClose);
    btnRow->addWidget(m_btnSave);

    mainLay->addLayout(btnRow);

    m_configPath = findConfigPath();
    m_settingsPath = findSettingsPath();
    loadGlobalSettings();
    loadSettings();
}

SettingsDialog::~SettingsDialog() {}

void SettingsDialog::applyDialogStyle() {
    setStyleSheet(R"(
        QDialog {
            background-color: #252526;
            color: #d4d4d4;
        }
        QLineEdit {
            background-color: #3c3c3c;
            color: #d4d4d4;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
        }
        QLineEdit:focus {
            border: 1px solid #007acc;
        }
        QComboBox {
            background-color: #3c3c3c;
            color: #d4d4d4;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
        }
    )");
}

QString SettingsDialog::findConfigPath() {
    // Cari folder moko_config secara rekursif ke atas dari application dir
    QDir dir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 5; ++i) {
        if (dir.exists("moko_config")) {
            QString path = dir.absoluteFilePath("moko_config/api_keys.json");
            qDebug() << "[SettingsDialog] Found config path at:" << path;
            return path;
        }
        if (dir.exists("api_keys.json")) {
            QString path = dir.absoluteFilePath("api_keys.json");
            qDebug() << "[SettingsDialog] Found config path at:" << path;
            return path;
        }
        if (!dir.cdUp()) break;
    }
    // Fallback ke relative path
    return "moko_config/api_keys.json";
}

QString SettingsDialog::findSettingsPath() {
    QDir dir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 5; ++i) {
        if (dir.exists("moko_config")) {
            QString path = dir.absoluteFilePath("moko_config/moko_settings.json");
            qDebug() << "[SettingsDialog] Found settings path at:" << path;
            return path;
        }
        if (!dir.cdUp()) break;
    }
    return "moko_config/moko_settings.json";
}

void SettingsDialog::loadGlobalSettings() {
    QFile file(m_settingsPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        // File doesn't exist yet — leave defaults
        updateLocalLLMStatus();
        return;
    }
    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        updateLocalLLMStatus();
        return;
    }

    QJsonObject obj = doc.object();

    // System Mode
    QString mode = obj.value("system_mode").toString("agent");
    if (mode == "rotation") {
        m_radioRotation->setChecked(true);
    } else {
        m_radioAgent->setChecked(true);
    }

    // Local LLM
    bool localEnabled = obj.value("local_llm_enabled").toBool(false);
    m_chkLocalLLM->setChecked(localEnabled);

    updateLocalLLMStatus();
}

void SettingsDialog::saveGlobalSettings() {
    QString mode = m_radioRotation->isChecked() ? "rotation" : "agent";

    QJsonObject obj;
    obj["system_mode"]      = mode;
    obj["local_llm_enabled"] = m_chkLocalLLM->isChecked();

    QJsonDocument doc(obj);
    QFile file(m_settingsPath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        file.write(doc.toJson());
        file.close();
    }
}

// ── Local LLM Slots ───────────────────────────────────────────────────────────

void SettingsDialog::onLocalLLMToggled(bool /*checked*/) {
    updateLocalLLMStatus();
}

void SettingsDialog::onModeChanged() {
    updateLocalLLMStatus();
}

void SettingsDialog::updateLocalLLMStatus() {
    bool localOn = m_chkLocalLLM->isChecked();

    // Determine if any external API row is marked "Active?"
    bool hasActiveAPI = false;
    for (int r = 0; r < m_table->rowCount(); ++r) {
        QCheckBox* cbActive = qobject_cast<QCheckBox*>(m_table->cellWidget(r, 1));
        if (cbActive && cbActive->isChecked()) {
            hasActiveAPI = true;
            break;
        }
    }

    if (!localOn) {
        m_lblLocalStatus->setText("⭕  Status: Tidak Aktif");
        m_lblLocalStatus->setStyleSheet(
            "color: #555; font-size: 11px; font-weight: 600; "
            "background: rgba(0,0,0,0.3); border-radius: 4px; padding: 3px 10px;"
        );
    } else if (localOn && hasActiveAPI) {
        m_lblLocalStatus->setText("📚  Status: Belajar (Learning Mode)");
        m_lblLocalStatus->setStyleSheet(
            "color: #ffd700; font-size: 11px; font-weight: 600; "
            "background: rgba(255,215,0,0.08); border: 1px solid rgba(255,215,0,0.3); "
            "border-radius: 4px; padding: 3px 10px;"
        );
    } else {
        // localOn && !hasActiveAPI
        m_lblLocalStatus->setText("⚡  Status: Mandor + Eksekutor Otonom (4-Phase)");
        m_lblLocalStatus->setStyleSheet(
            "color: #00e6ff; font-size: 11px; font-weight: 600; "
            "background: rgba(0,230,255,0.08); border: 1px solid rgba(0,230,255,0.3); "
            "border-radius: 4px; padding: 3px 10px;"
        );
    }
}


void SettingsDialog::loadSettings() {
    QFile file(m_configPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isArray()) return;

    QJsonArray arr = doc.array();
    for (int i = 0; i < arr.size(); ++i) {
        QJsonObject obj = arr[i].toObject();
        addRow();
        int row = m_table->rowCount() - 1;

        // Mandor checkbox
        QCheckBox* cb = qobject_cast<QCheckBox*>(m_table->cellWidget(row, 0));
        if (cb) cb->setChecked(obj["is_mandor"].toBool());

        // Active checkbox
        QCheckBox* cbActive = qobject_cast<QCheckBox*>(m_table->cellWidget(row, 1));
        if (cbActive) cbActive->setChecked(obj.contains("enabled") ? obj["enabled"].toBool() : true);

        // Name
        QLineEdit* nameEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(row, 2));
        if (nameEdit) nameEdit->setText(obj["name"].toString());

        // Provider
        QComboBox* provCombo = qobject_cast<QComboBox*>(m_table->cellWidget(row, 3));
        if (provCombo) provCombo->setCurrentText(obj["provider"].toString());

        // Model Name
        QLineEdit* modelEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(row, 4));
        if (modelEdit) modelEdit->setText(obj["model_name"].toString());

        // API Key(s) (joined by comma if array)
        QLineEdit* keyEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(row, 5));
        if (keyEdit) {
            QJsonArray keysArr = obj["api_keys"].toArray();
            QStringList keys;
            for (int k = 0; k < keysArr.size(); ++k) {
                keys << keysArr[k].toString();
            }
            keyEdit->setText(keys.join(", "));
        }

        // API Base
        QLineEdit* baseEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(row, 6));
        if (baseEdit) baseEdit->setText(obj["api_base"].toString());
    }
}

void SettingsDialog::addRow() {
    int row = m_table->rowCount();
    m_table->insertRow(row);

    // Mandor Checkbox
    QCheckBox* cb = new QCheckBox(this);
    cb->setStyleSheet("margin-left: 30px;");
    connect(cb, &QCheckBox::clicked, this, &SettingsDialog::onCellWidgetChanged);
    m_table->setCellWidget(row, 0, cb);

    // Active Checkbox
    QCheckBox* cbActive = new QCheckBox(this);
    cbActive->setStyleSheet("margin-left: 20px;");
    cbActive->setChecked(true);
    m_table->setCellWidget(row, 1, cbActive);

    // Name edit
    QLineEdit* nameEdit = new QLineEdit(this);
    m_table->setCellWidget(row, 2, nameEdit);

    // Provider combo
    QComboBox* provCombo = new QComboBox(this);
    provCombo->addItems({"gemini", "openai", "cloudflare", "custom"});
    m_table->setCellWidget(row, 3, provCombo);

    // Model edit
    QLineEdit* modelEdit = new QLineEdit(this);
    m_table->setCellWidget(row, 4, modelEdit);

    // Key edit
    QLineEdit* keyEdit = new QLineEdit(this);
    m_table->setCellWidget(row, 5, keyEdit);

    // Base URL edit
    QLineEdit* baseEdit = new QLineEdit(this);
    m_table->setCellWidget(row, 6, baseEdit);
}

void SettingsDialog::removeSelectedRow() {
    int row = m_table->currentRow();
    if (row >= 0) {
        m_table->removeRow(row);
    }
}

void SettingsDialog::onCellWidgetChanged() {
    // Ensure only one checkbox is checked for Mandor role
    QCheckBox* senderCb = qobject_cast<QCheckBox*>(sender());
    if (!senderCb || !senderCb->isChecked()) return;

    for (int r = 0; r < m_table->rowCount(); ++r) {
        QCheckBox* cb = qobject_cast<QCheckBox*>(m_table->cellWidget(r, 0));
        if (cb && cb != senderCb) {
            cb->setChecked(false);
        }
    }
}

void SettingsDialog::saveAndClose() {
    QJsonArray arr;

    for (int r = 0; r < m_table->rowCount(); ++r) {
        QCheckBox* cb = qobject_cast<QCheckBox*>(m_table->cellWidget(r, 0));
        QCheckBox* cbActive = qobject_cast<QCheckBox*>(m_table->cellWidget(r, 1));
        QLineEdit* nameEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(r, 2));
        QComboBox* provCombo = qobject_cast<QComboBox*>(m_table->cellWidget(r, 3));
        QLineEdit* modelEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(r, 4));
        QLineEdit* keyEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(r, 5));
        QLineEdit* baseEdit = qobject_cast<QLineEdit*>(m_table->cellWidget(r, 6));

        if (!nameEdit || nameEdit->text().trimmed().isEmpty()) continue;

        QJsonObject obj;
        obj["is_mandor"] = cb ? cb->isChecked() : false;
        obj["enabled"] = cbActive ? cbActive->isChecked() : true;
        obj["name"] = nameEdit->text().trimmed();
        obj["provider"] = provCombo ? provCombo->currentText() : "gemini";
        obj["model_name"] = modelEdit ? modelEdit->text().trimmed() : "";
        obj["api_base"] = baseEdit ? baseEdit->text().trimmed() : "";

        // Parse keys array
        QJsonArray keysArr;
        if (keyEdit) {
            QStringList keys = keyEdit->text().split(',', Qt::SkipEmptyParts);
            for (const QString& k : keys) {
                keysArr.append(k.trimmed());
            }
        }
        obj["api_keys"] = keysArr;

        arr.append(obj);
    }

    QJsonDocument doc(arr);
    QFile file(m_configPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Cannot write to config file:\n" + m_configPath);
        return;
    }

    file.write(doc.toJson());
    file.close();

    // Save global system mode setting
    saveGlobalSettings();

    QString modeLabel = m_radioRotation->isChecked()
        ? "🔄 API Rotation Mode"
        : "🤖 Agent AI Mode";

    QMessageBox::information(this, "Success",
        QString("Configuration saved!\n\nSystem Mode: %1\nAPI Config: %2\nSettings: %3")
            .arg(modeLabel, m_configPath, m_settingsPath));
    accept();
}
