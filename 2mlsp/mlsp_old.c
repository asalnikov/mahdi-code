/* To compile: gcc -shared -o mlsp.so mlsp.c -I /home/user/диплом/\(1\)\ slurm-modify/slurm-20.02.1-modify/ -fPIC */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/resource.h>
#include <slurm/spank.h>


// Work with predict-time-server
static int mlsp_ac;
static char** mlsp_av;

void json_getstr (char* json_str) {
    char str1[] = "{\"userId\":\"1\", \"username\": \"fizz bizz\"}";
    strcpy(json_str, str1);
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
    system (curl_str); /// обработать ошибки
    return 0;
}


// Work with spank

SPANK_PLUGIN(mlsp, 1);

static int _null_func (int val, const char *optarg, int remote) {
    return (0);
}

struct spank_option spank_options[] = {
    { "predict-time",
      "",
      "Predict time of running the task",
      0,
      0,
      (spank_opt_cb_f) _null_func // or null_func
    },
    SPANK_OPTIONS_TABLE_END
};


int slurm_spank_init (spank_t sp, int ac, char** av) {
    // Если не в srun / sbatch, то просто выйдем из функции и продолжится работа slurm, иначе предскажем время и выйдем
    //if (!( (spank_context () == S_CTX_LOCAL) || (spank_context () == S_CTX_ALLOCATOR) ))
        return (0);
    //_exit(predicttime(ac, av));
}

int slurm_spank_task_post_fork (spank_t sp, int ac, char **av) {
    _exit(predicttime(ac, av));
}
