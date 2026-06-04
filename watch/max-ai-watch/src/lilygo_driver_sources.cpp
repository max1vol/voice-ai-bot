// PlatformIO's Arduino library builder does not compile this library's nested
// driver folders by default when the library is consumed through lib_extra_dirs.
// Pull in only the driver implementations used by this firmware.

#include "../../TTGO_TWatch_Library/src/drive/i2c/i2c_bus.cpp"
#include "../../TTGO_TWatch_Library/src/drive/axp/axp20x.cpp"
#include "../../TTGO_TWatch_Library/src/drive/rtc/pcf8563.cpp"
#include "../../TTGO_TWatch_Library/src/drive/fx50xx/focaltech.cpp"
#include "../../TTGO_TWatch_Library/src/drive/bma423/bma4.c"
#include "../../TTGO_TWatch_Library/src/drive/bma423/bma423.c"
#include "../../TTGO_TWatch_Library/src/drive/bma423/bma.cpp"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/TFT_eSPI.cpp"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/glcdfont.c"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/Font16.c"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/Font32rle.c"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/Font64rle.c"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/Font72rle.c"
#include "../../TTGO_TWatch_Library/src/libraries/TFT_eSPI/Fonts/Font7srle.c"
