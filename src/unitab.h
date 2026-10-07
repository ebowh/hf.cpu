/* unitab.h - Unicode property tables (inversion lists), see tools/gen-unicode.pl. */
#ifndef HFC_UNITAB_H
#define HFC_UNITAB_H

#include <stddef.h>
#include <stdint.h>

extern const uint32_t hfc_uni_letter[];  extern const size_t hfc_uni_letter_n;   /* \p{L} */
extern const uint32_t hfc_uni_number[];  extern const size_t hfc_uni_number_n;   /* \p{N} */
extern const uint32_t hfc_uni_space[];   extern const size_t hfc_uni_space_n;    /* \s (White_Space) */

/* 1 if cp is in the set described by inversion list tab[0..n). */
static inline int hfc_uni_in(const uint32_t *tab, size_t n, uint32_t cp)
{
    size_t lo = 0, hi = n;                    /* count of entries <= cp */
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tab[mid] <= cp) lo = mid + 1; else hi = mid;
    }
    return (int)(lo & 1);
}

#define HFC_IS_LETTER(cp) hfc_uni_in(hfc_uni_letter, hfc_uni_letter_n, (cp))
#define HFC_IS_NUMBER(cp) hfc_uni_in(hfc_uni_number, hfc_uni_number_n, (cp))
#define HFC_IS_SPACE(cp)  hfc_uni_in(hfc_uni_space, hfc_uni_space_n, (cp))

#endif
