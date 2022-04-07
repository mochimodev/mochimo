/**
 * @private
 * @headerfile trigg.h <trigg.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TRIGG_C
#define MOCHIMO_TRIGG_C


#include "trigg.h"
#include <string.h>  /* for memory handling */

/**
 * @private
 * Dictionary and semantic grammar reference.
*/
static DICT Dict[MAXDICT] = {
/* Adverbs and function words */
   { "NIL", 0 },
   { "\n", F_OP },
   { "\b:", F_OP },
   { "\b--", F_OP },
   { "like", F_OP },
   { "a", F_OP },
   { "the", F_OP },
   { "of", F_OP },
   { "no", F_OP },
   { "\bs", F_OP },
   { "after", F_OP },
   { "before", F_OP },
/* Prepositions */
   { "at", F_PREP },
   { "in", F_PREP },
   { "on", F_PREP },
   { "under", F_PREP },
   { "above", F_PREP },
   { "below", F_PREP },
/* Verbs - intransitive ING and MOTION */
   { "arriving", F_ING | F_MOTION },
   { "departing", F_ING | F_MOTION },
   { "going", F_ING | F_MOTION },
   { "coming", F_ING | F_MOTION },
   { "creeping", F_ING | F_MOTION },
   { "dancing", F_ING | F_MOTION },
   { "riding", F_ING | F_MOTION },
   { "strutting", F_ING | F_MOTION },
   { "leaping", F_ING | F_MOTION },
   { "leaving", F_ING | F_MOTION },
   { "entering", F_ING | F_MOTION },
   { "drifting", F_ING | F_MOTION },
   { "returning", F_ING | F_MOTION },
   { "rising", F_ING | F_MOTION },
   { "falling", F_ING | F_MOTION },
   { "rushing", F_ING | F_MOTION },
   { "soaring", F_ING | F_MOTION },
   { "travelling", F_ING | F_MOTION },
   { "turning", F_ING | F_MOTION },
   { "singing", F_ING | F_MOTION },
   { "walking", F_ING | F_MOTION },
/* Verbs - intransitive ING */
   { "crying", F_ING },
   { "weeping", F_ING },
   { "lingering", F_ING },
   { "pausing", F_ING },
   { "shining", F_ING },
/* --- motion intransitive infinitive */
   { "fall", F_INF | F_MOTION },
   { "flow", F_INF | F_MOTION },
   { "wander", F_INF | F_MOTION },
   { "disappear", F_INF | F_MOTION },
/* --- intransitive infinitive */
   { "wait", F_INF },
   { "bloom", F_INF },
   { "doze", F_INF },
   { "dream", F_INF },
   { "laugh", F_INF },
   { "meditate", F_INF },
   { "listen", F_INF },
   { "sing", F_INF },
   { "decay", F_INF },
   { "cling", F_INF },
   { "grow", F_INF },
   { "forget", F_INF },
   { "remain", F_INF },
/* Adjectives - physical */
/* valences (e) based on Osgood's evaluation factor */
   { "arid", F_ADJ },
   { "abandoned", F_ADJ },
   { "aged", F_ADJ },
   { "ancient", F_ADJ },
   { "full", F_ADJ },
   { "glorious", F_ADJ },
   { "good", F_ADJ },
   { "beautiful", F_ADJ },
   { "first", F_ADJ },
   { "last", F_ADJ },
   { "forsaken", F_ADJ },
   { "sad", F_ADJ },
   { "mandarin", F_ADJ },
   { "naked", F_ADJ },
   { "nameless", F_ADJ },
   { "old", F_ADJ },
/* Ambient adjectives */
   { "quiet", F_ADJ | F_AMB },
   { "peaceful", F_ADJ },
   { "still", F_ADJ },
   { "tranquil", F_ADJ },
   { "bare", F_ADJ },
/* Time interval adjectives or nouns */
   { "evening", F_ADJ | F_TIMED },
   { "morning", F_ADJ | F_TIMED },
   { "afternoon", F_ADJ | F_TIMED },
   { "spring", F_ADJ | F_TIMEY },
   { "summer", F_ADJ | F_TIMEY },
   { "autumn", F_ADJ | F_TIMEY },
   { "winter", F_ADJ | F_TIMEY },
/* Adjectives - physical */
   { "broken", F_ADJ },
   { "thick", F_ADJ },
   { "thin", F_ADJ },
   { "little", F_ADJ },
   { "big", F_ADJ },
/* Physical + ambient adjectives */
   { "parched", F_ADJ | F_AMB },
   { "withered", F_ADJ | F_AMB },
   { "worn", F_ADJ | F_AMB },
/* Physical adj -- material things */
   { "soft", F_ADJ },
   { "bitter", F_ADJ },
   { "bright", F_ADJ },
   { "brilliant", F_ADJ },
   { "cold", F_ADJ },
   { "cool", F_ADJ },
   { "crimson", F_ADJ },
   { "dark", F_ADJ },
   { "frozen", F_ADJ },
   { "grey", F_ADJ },
   { "hard", F_ADJ },
   { "hot", F_ADJ },
   { "scarlet", F_ADJ },
   { "shallow", F_ADJ },
   { "sharp", F_ADJ },
   { "warm", F_ADJ },
   { "close", F_ADJ },
   { "calm", F_ADJ },
   { "cruel", F_ADJ },
   { "drowned", F_ADJ },
   { "dull", F_ADJ },
   { "dead", F_ADJ },
   { "sick", F_ADJ },
   { "deep", F_ADJ },
   { "fast", F_ADJ },
   { "fleeting", F_ADJ },
   { "fragrant", F_ADJ },
   { "fresh", F_ADJ },
   { "loud", F_ADJ },
   { "moonlit", F_ADJ | F_AMB },
   { "sacred", F_ADJ },
   { "slow", F_ADJ },
/* Nouns top-level */
/* Humans */
   { "traveller", F_NS },
   { "poet", F_NS },
   { "beggar", F_NS },
   { "monk", F_NS },
   { "warrior", F_NS },
   { "wife", F_NS },
   { "courtesan", F_NS },
   { "dancer", F_NS },
   { "daemon", F_NS },
/* Animals */
   { "frog", F_NS },
   { "hawks", F_NPL },
   { "larks", F_NPL },
   { "cranes", F_NPL },
   { "crows", F_NPL },
   { "ducks", F_NPL },
   { "birds", F_NPL },
   { "skylark", F_NS },
   { "sparrows", F_NPL },
   { "minnows", F_NPL },
   { "snakes", F_NPL },
   { "dog", F_NS },
   { "monkeys", F_NPL },
   { "cats", F_NPL },
   { "cuckoos", F_NPL },
   { "mice", F_NPL },
   { "dragonfly", F_NS },
   { "butterfly", F_NS },
   { "firefly", F_NS },
   { "grasshopper", F_NS },
   { "mosquitos", F_NPL },
/* Plants */
   { "trees", F_NPL | F_IN | F_AT },
   { "roses", F_NPL },
   { "cherries", F_NPL },
   { "flowers", F_NPL },
   { "lotuses", F_NPL },
   { "plums", F_NPL },
   { "poppies", F_NPL },
   { "violets", F_NPL },
   { "oaks", F_NPL | F_AT },
   { "pines", F_NPL | F_AT },
   { "chestnuts", F_NPL },
   { "clovers", F_NPL },
   { "leaves", F_NPL },
   { "petals", F_NPL },
   { "thorns", F_NPL },
   { "blossoms", F_NPL },
   { "vines", F_NPL },
   { "willows", F_NPL },
/* Things */
   { "mountain", F_NS | F_AT | F_ON },
   { "moor", F_NS | F_AT | F_ON | F_IN },
   { "sea", F_NS | F_AT | F_ON | F_IN },
   { "shadow", F_NS | F_IN },
   { "skies", F_NPL | F_IN },
   { "moon", F_NS },
   { "star", F_NS },
   { "stone", F_NS },
   { "cloud", F_NS },
   { "bridge", F_NS | F_ON | F_AT },
   { "gate", F_NS | F_AT },
   { "temple", F_NS | F_IN | F_AT },
   { "hovel", F_NS | F_IN | F_AT },
   { "forest", F_NS | F_IN | F_AT },
   { "grave", F_NS | F_IN | F_AT | F_ON },
   { "stream", F_NS | F_IN | F_AT | F_ON },
   { "pond", F_NS | F_IN | F_AT | F_ON },
   { "island", F_NS | F_ON | F_AT },
   { "bell", F_NS },
   { "boat", F_NS | F_IN | F_ON },
   { "sailboat", F_NS | F_IN | F_ON },
   { "bon fire", F_NS | F_AT },
   { "straw mat", F_NS | F_ON },
   { "cup", F_NS | F_IN },
   { "nest", F_NS | F_IN },
   { "sun", F_NS | F_IN },
   { "village", F_NS | F_IN },
   { "tomb", F_NS | F_IN | F_AT },
   { "raindrop", F_NS | F_IN },
   { "wave", F_NS | F_IN },
   { "wind", F_NS | F_IN },
   { "tide", F_NS | F_IN | F_AT },
   { "fan", F_NS },
   { "hat", F_NS },
   { "sandal", F_NS },
   { "shroud", F_NS },
   { "pole", F_NS },
/* Mass - substance */
   { "water", F_ON | F_IN | F_MASS | F_AMB },
   { "air", F_ON | F_IN | F_MASS | F_AMB },
   { "mud", F_ON | F_IN | F_MASS | F_AMB },
   { "rain", F_IN | F_MASS | F_AMB },
   { "thunder", F_IN | F_MASS | F_AMB },
   { "ice", F_ON | F_IN | F_MASS | F_AMB },
   { "snow", F_ON | F_IN | F_MASS | F_AMB },
   { "salt", F_ON | F_IN | F_MASS },
   { "hail", F_IN | F_MASS | F_AMB },
   { "mist", F_IN | F_MASS | F_AMB },
   { "dew", F_IN | F_MASS | F_AMB },
   { "foam", F_IN | F_MASS | F_AMB },
   { "frost", F_IN | F_MASS | F_AMB },
   { "smoke", F_IN | F_MASS | F_AMB },
   { "twilight", F_IN | F_AT | F_MASS | F_AMB },
   { "earth", F_ON | F_IN | F_MASS },
   { "grass", F_ON | F_IN | F_MASS },
   { "bamboo", F_MASS },
   { "gold", F_MASS },
   { "grain", F_MASS },
   { "rice", F_MASS },
   { "tea", F_IN | F_MASS },
   { "light", F_IN | F_MASS | F_AMB },
   { "darkness", F_IN | F_MASS | F_AMB },
   { "firelight", F_IN | F_MASS | F_AMB },
   { "sunlight", F_IN | F_MASS | F_AMB },
   { "sunshine", F_IN | F_MASS | F_AMB },
/* Abstract nouns and acts */
   { "journey", F_NS | F_ON },
   { "serenity", F_MASS },
   { "dusk", F_TIMED },
   { "glow", F_NS },
   { "scent", F_NS },
   { "sound", F_NS },
   { "silence", F_NS },
   { "voice", F_NS },
   { "day", F_NS | F_TIMED },
   { "night", F_NS | F_TIMED },
   { "sunrise", F_NS | F_TIMED },
   { "sunset", F_NS | F_TIMED },
   { "midnight", F_NS | F_TIMED },
   { "equinox", F_NS | F_TIMEY },
   { "noon", F_NS | F_TIMED }
};  /* end Dict[] */

/**
 * @private
 * Case frames for the semantic grammar with a vibe inspired by Basho...
*/
static word32 Frame[NFRAMES][MAXH] = {
   {
      F_PREP, F_ADJ, F_MASS, S_NL,            /* on a quiet moor */
      F_NPL, S_NL,                            /* raindrops       */
      F_INF | F_ING                           /* fall            */
   },
   {
      F_PREP, F_MASS, S_NL,
      F_ADJ, F_NPL, S_NL,
      F_INF | F_ING
   },
   {
      F_PREP, F_TIMED, S_NL,
      F_ADJ, F_NPL, S_NL,
      F_INF | F_ING
   },
   {
      F_PREP, F_TIMED, S_NL,
      S_A, F_NS, S_NL,
      F_ING
   },
   {
      F_TIME, F_AMB, S_NL,                    /* morning mist      */
      F_PREP, S_A, F_ADJ, F_NS, S_MD, S_NL,   /* on a worn field-- */
      F_ADJ | F_ING                           /* red               */
   },
   {
      F_TIME, F_AMB, S_NL,
      F_ADJ, F_MASS, S_NL,
      F_ING
   },
   {
      F_TIME, F_MASS, S_NL,                   /* morning mist */
      F_INF, S_S, S_CO, S_NL,                 /* remains:     */
      F_AMB                                   /* smoke        */
   },
   {
      F_ING, F_PREP, S_A, F_ADJ, F_NS, S_NL,  /* arriving at a parched gate */
      F_MASS, F_ING, S_MD, S_NL,              /* mist rises--               */
      S_A, F_ADJ, F_NS                        /* a moonlit sandal           */
   },
   {
      F_ING, F_PREP, F_TIME, F_MASS, S_NL,
      F_MASS, F_ING, S_MD, S_NL,
      S_A, F_ADJ, F_NS
   },
   {
      S_A, F_NS, S_NL,                        /* a wife              */
      F_PREP, F_TIMED, F_MASS, S_MD, S_NL,    /* in afternoon mist-- */
      F_ADJ                                   /* sad                 */
   }, /* ! increment NFRAMES if adding more frames... */
};  /* end Frame[][] */

/* Z_* constant semantics array lengths are rounded up to the nearest
 * power-of-2 for efficient use in trigg_generate_fast(). The effect
 * of repeat filler values on subsequent results is negligible. */

static const word8 Z_ING[32] = {
   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34
};
static const word8 Z_INF[16] = {
   44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
};
static const word8 Z_INGINF[32] = {
   18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 36, 37, 38, 39, 40,
   41, 42, 44, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59
};
static const word8 Z_NS[64] = {
   129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 145, 149, 154,
   155, 156, 157, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
   188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
   202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 241,
   244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
static const word8 Z_NPL[32] = {
   139, 140, 141, 142, 143, 144, 146, 147, 148, 150, 151,
   153, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
   168, 169, 170, 171, 172, 173, 174, 175, 176, 181
};
static const word8 Z_MASS[32] = {
   214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
   225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
   236, 237, 238, 239, 240, 242, 214, 215, 216, 219
};
static const word8 Z_AMB[16] = {
   77, 94, 95, 96, 126, 214, 217, 218, 220,
   222, 223, 224, 225, 226, 227, 228
};
static const word8 Z_TIMED[8] = { 84, 243, 249, 250, 251, 252, 253, 255 };
static const word8 Z_TIME[16] = {
   82, 83, 84, 85, 86, 87, 88, 243, 249, 250, 251, 252, 253, 254, 255, 253
};
static const word8 Z_PREP[8] = { 12, 13, 14, 15, 16, 17, 12, 13 };
static const word8 Z_ADJ[64] = {
   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
   76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
   105, 107, 108, 109, 110, 112, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 123, 124, 125, 126, 127, 128
};
static const word8 Z_INGADJ[64] = {
   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34,
   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
   77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92
};

/**
 * Generate a tokenized haiku. Generates tokenized haiku into `out` using
 * pseudo-rng from rand16().
 * @param out Pointer to place tokenized haiku into
 * @note Ensure random haiku generation by using srand16() beforehand.
*/
void *trigg_generate(void *out)
{
   word32 *fp;
   word8 *tp;
   int j, widx;

   /* choose a random haiku frame to fill */
   fp = &Frame[rand16() % NFRAMES][0];
   for (j = 0, tp = (word8 *) out; j < MAXH; j++, fp++, tp++) {
      if (*fp == 0) {
         /* zero fill to end of available token space */
         *tp = 0;
         continue;
      }
      if (*fp & F_XLIT) {
         /* force S_* type semantic feature where required by frame */
         widx = *fp & 255;
      } else {
         do { /* randomly select next word suitable for frame */
            widx = rand16() & MAXDICT_M1;
         } while ((Dict[widx].fe & *fp) == 0);
      }
      *tp = (word8) widx;
   }

   return out;
}  /* end trigg_generate() */

/**
 * Generate a tokenized haiku (fast). Generates tokenized haiku into @a out
 * using pseudo-rng from rand16(). Seed with srand16() before use.
 * @param out Pointer to place tokenized haiku into
*/
void *trigg_generate_fast(void *out)
{
   /* create entropy, and seed pointers */
   word32 rnd32[2] = { 0, 0 };
   word16 *rnd16 = (word16 *) rnd32;
   word8 *seed = (word8 *) out;
   int j, zero_from;

   /* generate random value < 0x48023C40000 (4,948,402,372,608) */
   rnd16[0] = rand16(); rnd16[1] = rand16(); rnd16[2] = rand16();
   if (rnd16[2] > 0x480) rnd16[2] %= 0x481;
   if (rnd16[2] == 0x480) rnd16[1] %= 0x23C4;

   /* determine frame type from rnd value */
   if (rnd32[1] > 0x80 || rnd32[1] == 0x80 && rnd32[0] >= 0x23C40000) {
      /* Permutations remaining: 4,948,402,372,608 (4.9 Trillion) */
      /* Permutations this frame: 4,398,046,511,104 (1 << 42) */
	   seed[ 0] = Z_ING[(rnd32[0] & 31)];
	   seed[ 1] = Z_PREP[(rnd32[0] >> 5) & 7];
	   seed[ 2] = 5;
	   seed[ 3] = Z_ADJ[(rnd32[0] >> 8) & 63];
	   seed[ 4] = Z_NS[(rnd32[0] >> 14) & 63];
	   seed[ 5] = 1;
      seed[ 6] = Z_MASS[(rnd32[0] >> 19) & 31];
	   seed[ 7] = Z_ING[(rnd32[0] >> 24) & 31];
	   seed[ 8] = 3;
	   seed[ 9] = 1;
      seed[10] = 5;
      /* 32-bit entropy boundary exceeded... */
	   seed[11] = Z_ADJ[((rnd32[0] >> 29) | (rnd32[1] << 3)) & 63];
      /* ... 32-bit entropy boundary passed */
	   seed[12] = Z_NS[(rnd32[1] >> 3) & 63];
      /* zero from 13th-byte... */
      zero_from = 13;
   } else if (rnd32[1] > 0 || rnd32[0] >= 0x23C40000) {
      /* Permutations remaining: 550,355,861,504 (550 Billion) */
      /* Permutations this frame: 549,755,813,888 (1 << 39) */
	   seed[ 0] = Z_ING[(rnd32[0] & 31)];
	   seed[ 1] = Z_PREP[(rnd32[0] >> 5) & 7];
	   seed[ 2] = Z_TIME[(rnd32[0] >> 8) & 15];
	   seed[ 3] = Z_MASS[(rnd32[0] >> 12) & 31];
	   seed[ 4] = 1;
      seed[ 5] = Z_MASS[(rnd32[0] >> 17) & 31];
	   seed[ 6] = Z_ING[(rnd32[0] >> 22) & 31];
	   seed[ 7] = 3;
	   seed[ 8] = 1;
      seed[ 9] = 5;
      /* 32-bit entropy boundary exceeded... */
	   seed[10] = Z_ADJ[((rnd32[0] >> 27) | (rnd32[1] << 5)) & 63];
      /* ... 32-bit entropy boundary passed */
	   seed[11] = Z_NS[(rnd32[1] >> 1) & 63];
      /* zero from 12th-byte... */
      zero_from = 12;
   } else if (rnd32[0] >= 0x3C40000) {
      /* Permutations remaining: 600,047,616 (600 Million) */
      /* Permutations this frame: 536,870,912 (1 << 29) */
	   seed[ 0] = Z_TIME[(rnd32[0] & 15)];
	   seed[ 1] = Z_AMB[(rnd32[0] >> 4) & 15];
	   seed[ 2] = 1;
      seed[ 3] = Z_PREP[(rnd32[0] >> 8) & 7];
	   seed[ 4] = 5;
	   seed[ 5] = Z_ADJ[(rnd32[0] >> 11) & 63];
	   seed[ 6] = Z_NS[(rnd32[0] >> 17) & 63];
	   seed[ 7] = 3;
	   seed[ 8] = 1;
      seed[ 9] = Z_INGADJ[(rnd32[0] >> 23) & 63];
      /* zero from 10th-byte... */
      zero_from = 10;
   } else if (rnd32[0] >= 0x2C40000) {
      /* Permutations remaining: 63,176,704 (63 Million) */
      /* Permutations this frame: 16,777,216 (1 << 24) */
	   seed[ 0] = Z_TIME[(rnd32[0] & 15)];
	   seed[ 1] = Z_AMB[(rnd32[0] >> 4) & 15];
	   seed[ 2] = 1;
      seed[ 3] = Z_ADJ[(rnd32[0] >> 8) & 63];
	   seed[ 4] = Z_MASS[(rnd32[0] >> 14) & 31];
	   seed[ 5] = 1;
      seed[ 6] = Z_ING[(rnd32[0] >> 19) & 31];
      /* zero from 7th-byte... */
      zero_from = 7;
   } else if (rnd32[0] >= 0x1C40000) {
      /* Permutations remaining: 46,399,488 (46 Million) */
      /* Permutations this frame: 16,777,216 (1 << 24) */
	   seed[ 0] = Z_PREP[(rnd32[0] & 7)];
	   seed[ 1] = Z_MASS[(rnd32[0] >> 3) & 31];
	   seed[ 2] = 1;
      seed[ 3] = Z_ADJ[(rnd32[0] >> 8) & 63];
	   seed[ 4] = Z_NPL[(rnd32[0] >> 14) & 31];
	   seed[ 5] = 1;
      seed[ 6] = Z_INGINF[(rnd32[0] >> 19) & 31];
      /* zero from 7th-byte... */
      zero_from = 7;
   } else if (rnd32[0] >= 0xC40000) {
      /* Permutations remaining: 29,622,272 (29 Million) */
      /* Permutations this frame: 16,777,216 (1 << 24) */
	   seed[ 0] = Z_PREP[rnd32[0] & 7];
	   seed[ 1] = Z_ADJ[(rnd32[0] >> 3) & 63];
	   seed[ 2] = Z_MASS[(rnd32[0] >> 9) & 31];
	   seed[ 3] = 1;
      seed[ 4] = Z_NPL[(rnd32[0] >> 14) & 31];
	   seed[ 5] = 1;
      seed[ 6] = Z_INGINF[(rnd32[0] >> 19) & 31];
      /* zero from 7th-byte... */
      zero_from = 7;
   } else if (rnd32[0] >= 0x440000) {
      /* Permutations remaining: 12,845,056 (12 Million) */
      /* Permutations this frame: 8,388,608 (1 << 23) */
	   seed[ 0] = 5;
	   seed[ 1] = Z_NS[(rnd32[0] & 63)];
	   seed[ 2] = 1;
      seed[ 3] = Z_PREP[(rnd32[0] >> 6) & 7];
	   seed[ 4] = Z_TIMED[(rnd32[0] >> 9) & 7];
	   seed[ 5] = Z_MASS[(rnd32[0] >> 12) & 31];
	   seed[ 6] = 3;
	   seed[ 7] = 1;
      seed[ 8] = Z_ADJ[(rnd32[0] >> 17) & 63];
      /* zero from 9th-byte... */
      zero_from = 9;
   } else if (rnd32[0] >= 0x40000) {
      /* Permutations remaining: 4,456,448 (4 Million) */
      /* Permutations this frame: 4,194,304 (1 << 22) */
	   seed[ 0] = Z_PREP[(rnd32[0] & 7)];
	   seed[ 1] = Z_TIMED[(rnd32[0] >> 3) & 7];
	   seed[ 2] = 1;
      seed[ 3] = Z_ADJ[(rnd32[0] >> 6) & 63];
	   seed[ 4] = Z_NPL[(rnd32[0] >> 12) & 31];
	   seed[ 5] = 1;
      seed[ 6] = Z_INGINF[(rnd32[0] >> 17) & 31];
      /* zero from 7th-byte... */
      zero_from = 7;
   } else if(rnd32[0] >= 0x20000) {
      /* Permutations remaining: 262,144 (262 Thousand) */
      /* Permutations this frame: 131,072 (1 << 17) */
	   seed[ 0] = Z_TIME[(rnd32[0] & 15)];
	   seed[ 1] = Z_MASS[(rnd32[0] >> 4) & 31];
	   seed[ 2] = 1;
      seed[ 3] = Z_INF[(rnd32[0] >> 9) & 15];
	   seed[ 4] = 9;
	   seed[ 5] = 2;
	   seed[ 6] = 1;
      seed[ 7] = Z_AMB[(rnd32[0] >> 13) & 15];
      /* zero from 8th-byte... */
      zero_from = 8;
   } else {
      /* Permutations remaining: 131,072 (131 Thousand) */
      /* Permutations this frame: 131,072 (1 << 17) */
	   seed[ 0] = Z_PREP[(rnd32[0] & 7)];
	   seed[ 1] = Z_TIMED[(rnd32[0] >> 3) & 7];
	   seed[ 2] = 1;
      seed[ 3] = 5;
	   seed[ 4] = Z_NS[(rnd32[0] >> 6) & 63];
	   seed[ 5] = 1;
      seed[ 6] = Z_ING[(rnd32[0] >> 12) & 31];
      /* zero from 7th-byte... */
      zero_from = 7;
   }

   /* clear remaining seed */
   memset(seed + zero_from, 0, 16 - zero_from);

   return out;
}  /* end trigg_generate_fast() */

/**
 * Expand a haiku to character format. It must have the correct syntax
 * and vibe.
 * @param nonce Pointer to tokenized haiku (nonce) to expand
 * @param haiku Pointer to character array to place expanded haiku
*/
char *trigg_expand(void *nonce, void *haiku)
{
   word8 *np, *bp, *bpe, *wp;
   int i;

   np = (word8 *) nonce;
   bp = (word8 *) haiku;
   bpe = bp + HAIKUCHARLEN;
   /* step through all nonce values */
   for (i = 0; i < MAXH; i++, np++) {
      if (*np == 0) break;
      /* place word from dictionary into bp */
      wp = Dict[*np].tok;
      while (*wp) *(bp++) = *(wp++);
      if (bp[-1] != '\n') *(bp++) = ' ';
   }
   /* zero fill remaining character space */
   i = (bpe - bp) & 3;
   while (i--) *(bp++) = 0;  /* 8-bit fill */
   while (bp < bpe) {
      *((word32 *) bp) = 0;
      bp += 4;
   }

   return (char *) haiku;
}  /* end trigg_expand() */

/**
 * Evaluate the TRIGG chain by using a heuristic estimate of the final
 * solution cost (Nilsson, 1971). Evaluate the relative distance within
 * the TRIGG chain to validate proof of work.
 * @param hash Pointer to hash to evaluate
 * @param diff Difficulty to evaluate hash against
 * @returns VEOK if passed, else VERROR on fail
*/
int trigg_eval(void *hash, word8 diff)
{
   word8 *bp, n;

   n = diff >> 3;
   /* coarse check required bytes are zero */
   for (bp = (word8 *) hash; n; n--) {
      if(*(bp++) != 0) return VERROR;
   }
   if ((diff & 7) == 0) return VEOK;
   /* fine check required bits are zero */
   if ((*bp & ~(0xff >> (diff & 7))) != 0) {
      return VERROR;
   }

   return VEOK;
}  /* end trigg_eval() */

/**
 * Check haiku syntax against semantic grammar. It must have the correct
 * syntax, semantics, and vibe.
 * @param nonce Pointer to tokenized haiku (nonce) to check
 * @returns VEOK on correct syntax, else VERROR if incorrect
*/
int trigg_syntax(void *nonce)
{
   word32 sf[MAXH], *fp;
   word8 *np;
   int j;

   /* load semantic frame associated with nonce */
   for (j = 0, np = (word8 *) nonce; j < MAXH; j++) sf[j] = Dict[np[j]].fe;
   /* check input for respective semantic features, use unification on sets. */
   for (fp = &Frame[0][0]; fp < &Frame[NFRAMES][0]; fp += MAXH) {
      for (j = 0; j < MAXH; j++) {
         if (fp[j] == 0) {
            if (sf[j] == 0) return VEOK;
            break;
         }
         if (fp[j] & F_XLIT) {
            if ((fp[j] & 255) != np[j]) break;
            continue;
         }
         if ((sf[j] & fp[j]) == 0) break;
      }
      if (j >= MAXH) return VEOK;
   }

   return VERROR;
}  /* end trigg_syntax() */

/**
 * Check proof of work. The haiku must be syntactically correct and have
 * the right vibe. Also, entropy MUST match difficulty.
 * If non-NULL, place final hash in `out` on success.
 * @param bt Pointer to block trailer to check
 * @param out Pointer to byte array to place the final hash (if non-NULL)
 * @returns Hash evaluation result as; VEOK on success, else VERROR
*/
int trigg_checkhash(BTRAILER *bt, void *out)
{
   SHA256_CTX ictx;
   word8 haiku[HAIKUCHARLEN], hash[SHA256LEN];

   /* check syntax, semantics, and vibe... */
   if (trigg_syntax(bt->nonce) == VERROR) return VERROR;
   if (trigg_syntax(bt->nonce + 16) == VERROR) return VERROR;
   /* re-linearise the haiku */
   trigg_expand(bt->nonce, haiku);
   /* obtain entropy */
   sha256_init(&ictx);
   sha256_update(&ictx, bt->mroot, SHA256LEN);
   sha256_update(&ictx, haiku, HAIKUCHARLEN);
   sha256_update(&ictx, bt->nonce + 16, 16);
   sha256_update(&ictx, bt->bnum, 8);
   sha256_final(&ictx, hash);
   /* pass final hash to `out` if not NULL */
   if (out != NULL) memcpy(out, hash, SHA256LEN);
   /* return evaluation */
   return trigg_eval(hash, bt->difficulty[0]);
}  /* end trigg_checkhash() */

/**
 * Prepare a TRIGG context for solving.
 * @param T Pointer to Trigg solving context
 * @param bt Pointer to block trailer with data to be solved
*/
void trigg_init(TRIGG_CTX *T, BTRAILER *bt)
{
   /* add merkle root and bnum to Tchain */
   memcpy(T->mroot, bt->mroot, SHA256LEN);
   memcpy(T->bnum, bt->bnum, 8);
   /* place required difficulty in diff */
   T->diff = bt->difficulty[0];
}  /* end trigg_init() */

/**
 * Try to solve proof of work with a tokenized haiku as nonce output.
 * Create the haiku inside the TRIGG chain using a semantic grammar
 * (Burton, 1976). The output must pass syntax checks, the entropy
 * check, and have the right vibe. Entropy is always preserved at
 * high difficulty levels. Place nonce into `out` on success.
 * @param T Pointer to Trigg solving context
 * @param out Pointer to byte array to place nonce (on solve)
 * @returns VEOK on success, else VERROR
*/
int trigg_solve(TRIGG_CTX *T, void *out)
{
   word8 hash[SHA256LEN];

   /* generate (full) nonce */
   trigg_generate(T->nonce2);
   trigg_generate(T->nonce1);
   /* expand shifted nonce into the TRIGG chain! */
   trigg_expand(T->nonce1, T->haiku);
   /* perform SHA256 hash on TRIGG chain */
   sha256(T, TCHAINLEN, hash);
   /* evaluate result against required difficulty */
   if (trigg_eval(hash, T->diff) == VEOK) {
      /* copy successful (full) nonce to `out` */
      word8 *bp = (word8 *) out;
      memcpy(bp, T->nonce1, 16);
      memcpy(bp + 16, T->nonce2, 16);
      return VEOK;
   }

   return VERROR;
}  /* end trigg_solve() */

/* end include guard */
#endif
