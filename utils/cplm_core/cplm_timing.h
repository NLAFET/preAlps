#ifndef CPLM_TIMING_H
#define CPLM_TIMING_H

#ifndef USE_CPALAMEM
/* Disable timing and instrumentation macro and functions when preAlps is compiled without cpalamem */
  #define CPLM_PUSH
  #define CPLM_POP
  #define CPLM_BEGIN_TIME
  #define CPLM_END_TIME
  #define CPLM_OPEN_TIMER
  #define CPLM_CLOSE_TIMER
  #define CPLM_TIC(a, b)
  #define CPLM_TAC(a)
  #define CPLM_SetEnv()
  #define CPLM_printTimer(a)
  enum {
  step1 = 1,
  step2,
  step3,
  step4,
  step5,
  step6,
  step7,
  step8,
  step9,
  step10,
  step11,
  step12,
  step13,
  step14,
  step15,
  step16,
  step17,
  step18,
  step19,
  step20,
  step21,
  step22,
  step23,
  step24,
  step25,
  step26,
  step27,
  step28,
  step29,
  step30
};
#endif

#endif
