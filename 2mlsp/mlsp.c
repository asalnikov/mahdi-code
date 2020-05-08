/*
    Compile: gcc -shared -o mlsp.so mlsp.c -I /home/user/диплом/\(1\)\ slurm-modify/slurm-20.02.1-modify/ -fPIC
    Test:
        python3 test.py
        sudo systemctl restart slurm*  (только при изменении plugstack.conf)
        srun --predict-time -l echo test
*/

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/resource.h>
#include <slurm/spank.h>

#include <src/common/plugstack.c>


// Work with predict-time-server

// ["s1", "s2", "s3"]
void json_getstr (char* json_str, int ac, char** av) {
    json_str[0] = '\0';
    strcat(json_str, "[");
    for (int i = 0; i < ac; ++i) {
        strcat(json_str, "\"");
        strcat(json_str, av[i]);
        strcat(json_str, "\"");
        if (i != ac - 1)
            strcat(json_str, ", ");
    }
    strcat(json_str, "]");
}

const char curl_headstr[] = "curl -H \"Content-Type: application/json\" -X POST http://localhost:4567/ -d ";

// curl -H "Content-Type: application/json" -X POST http://localhost:4567/ -d '{"userId":"1", "username": "fizz bizz"}'
void curl_getstr (char* curl_str, int ac, char** av) {
    curl_str[0] = '\0';
    strcat(curl_str, curl_headstr);
    strcat(curl_str, "\'");
    char json_str[BUFSIZ];
    json_getstr(json_str, ac, av);
    strcat(curl_str, json_str);
    strcat(curl_str, "\'");
}

int predicttime (int ac, char** av) {
    char curl_str[BUFSIZ];
    curl_getstr(curl_str, ac, av);
    //slurm_info(curl_str);
    system (curl_str); /// обработать ошибки
    return 0;
}


// Work with spank

SPANK_PLUGIN(mlsp, 1);

static bool has_predicttime_arg = false;

//static int predicttime_wrapper (int val, const char *optarg, int remote) {
void predicttime_wrapper (int ac, char** av) {
    predicttime(ac, av);
    _exit (0);
    //return 0;
}

static int has_predicttime_arg_func (int val, const char *optarg, int remote) {
    has_predicttime_arg = true;
    return 0;
}

struct spank_option spank_options[] = {
    { "predict-time",
      "",
      "Predict time of running the task",
      0,
      0,
      (spank_opt_cb_f) has_predicttime_arg_func
    },
    SPANK_OPTIONS_TABLE_END
};

bool true_spank_context() {
    return ( (spank_context () != S_CTX_LOCAL) || (spank_context () != S_CTX_ALLOCATOR) ) ? false : true;
}

int slurm_spank_init_post_opt(spank_t sp, int ac, char** av) {
    if ( !true_spank_context() || !has_predicttime_arg)
        return 0;

    pid_t srun_pid = getpid();
    char* cmdline_file_str[BUFSIZ];
    sprintf(cmdline_file_str, "/proc/%d/cmdline", (int)srun_pid);

    FILE* cmdline_file = fopen(cmdline_file_str, "r");
    /// dinamic cmdline
    char cmdline[BUFSIZ];
    int cmdline_len = fread(cmdline, 1, BUFSIZ, cmdline_file);

    int proc_ac = 0;
    char** proc_av;
    for (int i = 0; i < cmdline_len; ++i) {
        int c = i;
        if (cmdline[i] == '\0') {

        }
    }
    //printf("%d\n%s\n", cmdline_len, cmdline);

    predicttime_wrapper(proc_ac, proc_av);

    /// clear memory

    return 0;
}
