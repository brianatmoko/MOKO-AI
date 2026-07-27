#include <QApplication>
#include "moko_window.h"
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
    // Seed random number generator for hardware simulator
    srand(static_cast<unsigned int>(time(nullptr)));

    QApplication app(argc, argv);
    app.setApplicationName("Moko Native IDE");
    app.setApplicationVersion("6.0.0");

    MokoWindow window;
    window.show();

    return app.exec();
}
