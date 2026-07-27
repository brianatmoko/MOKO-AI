#ifndef HELPER_ENGINE_H
#define HELPER_ENGINE_H

#include <QString>
#include <QVector>

struct DiagnosticIssue {
    int line;
    int col;
    QString severity; // "ERROR" (salah) atau "WARNING" (kurang tepat)
    QString message;
    QString codeSnippet;
};

class HelperEngine {
public:
    HelperEngine();
    ~HelperEngine();

    // Jalankan lint & diagnosis lengkap pada kode
    QVector<DiagnosticIssue> analyzeCode(const QString& code);

    // Hasilkan prompt terkompresi berbasis COORD_2D focal zones untuk AI
    QString generateCompressedContext(const QString& code, const QString& filePath);
};

#endif // HELPER_ENGINE_H
