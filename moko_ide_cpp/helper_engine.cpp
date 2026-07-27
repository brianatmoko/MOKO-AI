#include "helper_engine.h"
#include <QStringList>
#include <QRegularExpression>
#include <QStack>
#include <cmath>

HelperEngine::HelperEngine() {}

HelperEngine::~HelperEngine() {}

QVector<DiagnosticIssue> HelperEngine::analyzeCode(const QString& code) {
    QVector<DiagnosticIssue> issues;
    QStringList lines = code.split('\n');

    // Regex patterns
    QRegularExpression importRe("^\\s*import\\s+([a-zA-Z0-9_]+)");
    QRegularExpression fromImportRe("^\\s*from\\s+([a-zA-Z0-9_]+)\\s+import\\s+([a-zA-Z0-9_]+)");
    QRegularExpression defRe("^\\s*(def|class|if|elif|else|for|while|try|except|with)\\b");
    QRegularExpression emptyExceptRe("^\\s*except\\s*:\\s*$");
    QRegularExpression passRe("^\\s*pass\\s*$");
    QRegularExpression printRe("\\bprint\\s*\\(");
    QRegularExpression semicolonRe(";");

    struct BracketInfo {
        QChar type;
        int line;
        int col;
    };
    QStack<BracketInfo> bracketStack;

    // Track imports
    struct ImportInfo {
        QString name;
        int line;
        bool used;
    };
    QVector<ImportInfo> imports;

    for (int i = 0; i < lines.size(); ++i) {
        QString line = lines[i];
        QString trimmed = line.trimmed();
        int lineNum = i + 1;

        if (trimmed.isEmpty() || trimmed.startsWith('#')) {
            continue;
        }

        // 1. Cek brackets balancing
        for (int col = 0; col < line.length(); ++col) {
            QChar ch = line.at(col);
            if (ch == '(' || ch == '[' || ch == '{') {
                bracketStack.push(BracketInfo{ch, lineNum, col + 1});
            } else if (ch == ')' || ch == ']' || ch == '}') {
                if (bracketStack.isEmpty()) {
                    issues.append(DiagnosticIssue{
                        lineNum, col + 1, "ERROR",
                        QString("Kelebihan penutup braket '%1' tanpa pembuka.").arg(ch),
                        trimmed
                    });
                } else {
                    BracketInfo open = bracketStack.top();
                    if ((ch == ')' && open.type == '(') ||
                        (ch == ']' && open.type == '[') ||
                        (ch == '}' && open.type == '{')) {
                        bracketStack.pop();
                    } else {
                        issues.append(DiagnosticIssue{
                            lineNum, col + 1, "ERROR",
                            QString("Ketidakcocokan braket: pembuka '%1' di baris %2 ditutup oleh '%3'.").arg(open.type).arg(open.line).arg(ch),
                            trimmed
                        });
                        bracketStack.pop();
                    }
                }
            }
        }

        // 2. Deteksi 'def/class/if...' tanpa titik dua ':'
        QRegularExpressionMatch defMatch = defRe.match(line);
        if (defMatch.hasMatch()) {
            if (!trimmed.endsWith(':') && !trimmed.contains('#')) {
                issues.append(DiagnosticIssue{
                    lineNum, line.length(), "ERROR",
                    QString("Pernyataan blok '%1' kekurangan tanda titik dua ':' di akhir baris.").arg(defMatch.captured(1)),
                    trimmed
                });
            }
        }

        // 3. Deteksi import dan simpan
        QRegularExpressionMatch importMatch = importRe.match(line);
        if (importMatch.hasMatch()) {
            imports.append(ImportInfo{importMatch.captured(1), lineNum, false});
        }
        QRegularExpressionMatch fromImportMatch = fromImportRe.match(line);
        if (fromImportMatch.hasMatch()) {
            imports.append(ImportInfo{fromImportMatch.captured(2), lineNum, false});
        }

        // 4. Deteksi suboptimal: 'except:' diikuti oleh 'pass'
        if (emptyExceptRe.match(trimmed).hasMatch()) {
            // Cek baris berikutnya apakah 'pass'
            if (i + 1 < lines.size()) {
                QString nextLine = lines[i + 1].trimmed();
                if (passRe.match(nextLine).hasMatch()) {
                    issues.append(DiagnosticIssue{
                        lineNum, 1, "WARNING",
                        "Blok except kosong (bare except + pass). Sebaiknya tangkap exception spesifik atau log kesalahan.",
                        trimmed + " " + nextLine
                    });
                }
            }
        }

        // 5. Deteksi suboptimal: print debugging
        if (printRe.match(line).hasMatch()) {
            issues.append(DiagnosticIssue{
                lineNum, 1, "WARNING",
                "Ditemukan fungsi print() untuk debugging. Untuk rilis produksi, gunakan logging module.",
                trimmed
            });
        }

        // 6. Deteksi suboptimal: penggunaan titik koma ';'
        if (semicolonRe.match(line).hasMatch()) {
            issues.append(DiagnosticIssue{
                lineNum, 1, "WARNING",
                "Penggunaan tanda titik koma ';' untuk menggabungkan beberapa pernyataan dalam satu baris tidak direkomendasikan.",
                trimmed
            });
        }
    }

    // Cek braket yang tidak ditutup
    while (!bracketStack.isEmpty()) {
        BracketInfo open = bracketStack.pop();
        issues.append(DiagnosticIssue{
            open.line, open.col, "ERROR",
            QString("Braket pembuka '%1' tidak ditutup.").arg(open.type),
            lines[open.line - 1].trimmed()
        });
    }

    // Cek unused imports
    for (const auto& imp : imports) {
        bool found = false;
        QRegularExpression wordRe(QString("\\b%1\\b").arg(imp.name));
        for (int i = 0; i < lines.size(); ++i) {
            if (i + 1 == imp.line) continue; // skip baris import itu sendiri
            if (wordRe.match(lines[i]).hasMatch()) {
                found = true;
                break;
            }
        }
        if (!found) {
            issues.append(DiagnosticIssue{
                imp.line, 1, "WARNING",
                QString("Library '%1' di-import tetapi tidak pernah digunakan di dalam kode.").arg(imp.name),
                lines[imp.line - 1].trimmed()
            });
        }
    }

    return issues;
}

QString HelperEngine::generateCompressedContext(const QString& code, const QString& filePath) {
    QVector<DiagnosticIssue> issues = analyzeCode(code);
    QStringList lines = code.split('\n');

    QString context;
    context.append("--- DEEPSEEK-OCR 2 CAUSAL VISUAL FLOW MAP ---\n");
    context.append(QString("File: %1 | Size: %2 lines\n\n").arg(filePath).arg(lines.size()));

    if (issues.isEmpty()) {
        context.append("[STATUS: CLEAN]\nKode tidak memiliki masalah sintaksis atau peringatan suboptimal.\n");
        return context;
    }

    context.append("[OPTICAL_ANCHOR - COORDINATE MAP]\n");
    
    // Kelompokkan masalah
    QVector<DiagnosticIssue> errors;
    QVector<DiagnosticIssue> warnings;
    for (const auto& issue : issues) {
        if (issue.severity == "ERROR") {
            errors.append(issue);
        } else {
            warnings.append(issue);
        }
    }

    // 1. Laporkan ERROR (salah)
    if (!errors.isEmpty()) {
        context.append("\n# BAGIAN YANG SALAH (CRITICAL ERRORS):\n");
        for (const auto& err : errors) {
            // Hasilkan Bounding Box fiktif COORD_2D=[xmin,ymin,xmax,ymax] untuk OCR simulator
            int y = err.line;
            int x = err.col;
            context.append(QString("COORD_2D=[%1,%2,%3,%4] -> L%5: %6\n")
                .arg(x).arg(y).arg(x + 10).arg(y)
                .arg(y).arg(err.message));
            context.append(QString("  `%1`\n").arg(err.codeSnippet));
        }
    }

    // 2. Laporkan WARNING (kurang tepat)
    if (!warnings.isEmpty()) {
        context.append("\n# BAGIAN KURANG TEPAT (SUBOPTIMAL WARNINGS):\n");
        for (const auto& warn : warnings) {
            int y = warn.line;
            int x = warn.col;
            context.append(QString("COORD_2D=[%1,%2,%3,%4] -> L%5: %6\n")
                .arg(x).arg(y).arg(x + 10).arg(y)
                .arg(y).arg(warn.message));
            context.append(QString("  `%1`\n").arg(warn.codeSnippet));
        }
    }

    context.append("\n[DECODER_DIRECTIVE]: Analisis visual selesai. Gunakan koordinat 2D untuk langsung memperbaiki baris terkait.");
    return context;
}
