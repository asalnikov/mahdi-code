/* To compile: gcc -shared -o mlsp.so mlsp.c -I /home/user/диплом/\(1\)\ slurm-modify/slurm-20.02.1-modify/ -fPIC */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/resource.h>
#include <slurm/spank.h>

/*
 * All spank plugins must define this macro for the
 * Slurm plugin loader.
 */
SPANK_PLUGIN(mlsp, 1);


static int _predicttime (int val, const char *optarg, int remote) {
    ///   как сюда передать аргументы srun?

    _exit(0);
}

/*
 *  Provide a --predict-time option to srun:
 */
struct spank_option predicttime_opt[] = {
    { "predict-time",
      "",
      "Predict time of running the task",
      0,
      0,
      (spank_opt_cb_f) _predicttime
    },
    SPANK_OPTIONS_TABLE_END
};

int slurm_spank_init (spank_t sp, int ac, char** av) {
    if !( (spank_context () == S_CTX_LOCAL) || (spank_context () == S_CTX_ALLOCATOR) )
        return (0);
    spank_option_register (sp, predicttime_opt); // add check
    return (0);
}
