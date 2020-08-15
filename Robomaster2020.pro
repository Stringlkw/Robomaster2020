# Configs
TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++14

#Libraries
unix: CONFIG += link_pkgconfig

#OpenCV
unix: PKGCONFIG += opencv

# 暂时还未用到CUDA
#CUDA
#unix:!macx: LIBS += -L$$PWD/../../../usr/local/cuda/lib64/-lcudart
#unix:!macx: LIBS += -L$$PWD/../../../usr/local/cuda/lib64/-lcuda
#unix:!macx: LIBS += -L$$PWD/../../../usr/local/cuda/lib64/-lcublas
#unix:!macx: LIBS += -L$$PWD/../../../usr/local/cuda/lib64/-lcurand
#INCLUDEPATH += $$PWD/../../../usr/local/cuda/include
#DEPENDPATH += $$PWD/../../../usr/local/cuda/include

#V4L2
unix:!macx: LIBS += -lpthread
unix:!macx: LIBS += -lv4l2


#Source and header files
HEADERS += \
#    General/singleton.hpp \
    General/opencv_extended.h \
    General/numeric_rm.h \
    General/General.h \
#    Driver/RMVideoCapture.hpp \
    Serials/Serial.h \
#    Main/ImgProdCons.h \
#    Pose/Predictor.hpp \
    Pose/AngleSolver.hpp \
    Armor/ArmorDetector.h \
    WindMill/WindMill.h \

SOURCES += \
    General/opencv_extended.cpp \
    General/numeric_rm.cpp \
#    Driver/RMVideoCapture.cpp \
    Serials/Serial.cpp \
#    Main/ImgProdCons.cpp \
#    Pose/Predictor.cpp \
    Pose/AngleSolver.cpp \
    Armor/ArmorDetector.cpp \
    WindMill/WindMill.cpp \
#   To test a sigular module, uncomment only one of the following
#    Armor/test_armor.cpp
#    Driver/test_camera.cpp
#    Serials/test_serials.cpp
#    Pose/test_angle_2.cpp
#    Pose/test_angle_1.cpp
#   To test a singular module with multi-thread, uncomment main.cpp
#   and one of the others
    Main/main.cpp \
#    Main/test_armor_temp.cpp
#    Main/test_producer.cpp
#    Main/test_armor.cpp
#    Main/test_armor_solver.cpp
#    Main/test_rune.cpp \
#    Main/test_armor_solver_serial.cpp
#     Main/test_recording_video.cpp
#    Main/test_sentry.cpp


DISTFILES += \
    SVM3.xml \
    red.avi \
    template/template1.jpg \
    template/template2.jpg \
    template/template3.jpg \
    template/template4.jpg \
    template/template5.jpg \
    template/template6.jpg \
    template/template7.jpg \
    template/template8.jpg


