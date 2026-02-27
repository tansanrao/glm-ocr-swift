#ifndef FPDFVIEW_H_
#define FPDFVIEW_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef const char* FPDF_BYTESTRING;
typedef void* FPDF_DOCUMENT;
typedef void* FPDF_PAGE;
typedef void* FPDF_BITMAP;

void FPDF_InitLibrary(void);
void FPDF_DestroyLibrary(void);

FPDF_DOCUMENT FPDF_LoadMemDocument64(const void* data_buf, size_t size, FPDF_BYTESTRING password);
void FPDF_CloseDocument(FPDF_DOCUMENT document);
int FPDF_GetPageCount(FPDF_DOCUMENT document);
FPDF_PAGE FPDF_LoadPage(FPDF_DOCUMENT document, int page_index);
void FPDF_ClosePage(FPDF_PAGE page);
float FPDF_GetPageWidthF(FPDF_PAGE page);
float FPDF_GetPageHeightF(FPDF_PAGE page);
unsigned long FPDF_GetLastError(void);

FPDF_BITMAP FPDFBitmap_CreateEx(int width, int height, int format, void* first_scan, int stride);
void FPDFBitmap_Destroy(FPDF_BITMAP bitmap);
void* FPDFBitmap_GetBuffer(FPDF_BITMAP bitmap);
int FPDFBitmap_GetStride(FPDF_BITMAP bitmap);
void FPDFBitmap_FillRect(FPDF_BITMAP bitmap, int left, int top, int width, int height, unsigned int color);

void FPDF_RenderPageBitmap(
    FPDF_BITMAP bitmap,
    FPDF_PAGE page,
    int start_x,
    int start_y,
    int size_x,
    int size_y,
    int rotate,
    int flags
);

enum {
    FPDFBitmap_Unknown = 0,
    FPDFBitmap_Gray = 1,
    FPDFBitmap_BGR = 2,
    FPDFBitmap_BGRx = 3,
    FPDFBitmap_BGRA = 4
};

enum {
    FPDF_ANNOT = 0x01
};

#ifdef __cplusplus
}
#endif

#endif
