/*
    Compile: gcc -shared -o mlsp.so mlsp.c -I /home/user/диплом/\(1\)\ slurm-modify/slurm-20.02.1-modify/ -fPIC -lcurl
    Test:
        Modify /etc/slurm???/plugstack.conf: add "required /home/user/диплом/mahdi-code/2mlsp/mlsp.so"
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
#include <curl/curl.h>
#include <slurm/spank.h>

#include <src/common/plugstack.c> // потом лучше убрать


// Work with predict-time-server

// ["s1", "s2", "s3"]
void json_getstr (char** json_str, int ac, char** av) {
    int json_len = 2;
    char* jsons = (char*)malloc(json_len * sizeof(char));
    jsons[0] = '\0';
    strcat(jsons, "[");
    for (int i = 0; i < ac; ++i) {
        int av_i_len = strlen(av[i]);
        jsons = (char*)realloc(jsons, (json_len += av_i_len + 4)*sizeof(char));
        strcat(jsons, "\"");
        strcat(jsons, av[i]);
        strcat(jsons, "\"");
        if (i != ac - 1)
            strcat(jsons, ", ");
    }
    strcat(jsons, "]");
    *json_str = jsons;
}

// curl -H "Content-Type: application/json" -X POST http://localhost:4567/ -d '{"userId":"1", "username": "fizz bizz"}'
size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) {
    strncpy((char*)userdata, ptr, BUFSIZ);
    ((char*)userdata)[BUFSIZ - 1] = '\0';
    return size * nmemb;
}

int predicttime (int ac, char** av) {
    CURL* curl = curl_easy_init();
    if (!curl)
        return 1;

    curl_global_init(CURL_GLOBAL_ALL);

    // http://localhost:4567/
    const char config_tag[] = "server_url ";
    FILE* config_file = fopen("mlsp.conf", "r");
    if (config_file == NULL) {
        slurm_info("Configuration file doesn't exist / allowed");
        return 2;
    }
    char config_str[BUFSIZ];
    config_str[0] = '\0';
    while (!feof(config_file)) {
        fgets(config_str, BUFSIZ, config_file);
        if (!strncmp(config_str, config_tag, sizeof(config_tag)))
            break;
    }
    fclose(config_file);
    if (config_str[0] == '\0') {
        slurm_info("Error of reading configuration file");
        return 3;
    }
    config_str[strlen(config_str) - 2] = '\0';
    //slurm_info("Connect to server: %s", config_str + sizeof(config_tag) - 1);
    curl_easy_setopt(curl, CURLOPT_URL, config_str + sizeof(config_tag) - 1);


    // -X POST -d '["s1", "s2", "s3"]'
    char* json_str;
    json_getstr(&json_str, ac, av);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);

    // -H "Content-Type: application/json"
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // get answer init
    char ans[BUFSIZ];
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, ans);

    // get answer
    CURLcode res = curl_easy_perform(curl);
    if (res == CURLE_OK)
        slurm_info("Predict time of running: %s", ans);
    else
        slurm_info("Error: %s", curl_easy_strerror(res));

    // clearing memory
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    free(json_str);

    return (res == CURLE_OK) ? 0 : 4;
}


// Work with spank

SPANK_PLUGIN(mlsp, 1);

static bool has_predicttime_arg = false;

static int has_predicttime_arg_func (int val, const char *optarg, int remote) {
    has_predicttime_arg = true;
    return 0;
}

struct spank_option predicttime_opt[] = {
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
    return ( (spank_context () == S_CTX_LOCAL) || (spank_context () == S_CTX_ALLOCATOR) );
}

int slurm_spank_init(spank_t sp, int ac, char** av) {
    if ( true_spank_context() )
        spank_option_register(sp, predicttime_opt);
    return 0;
}

int slurm_spank_init_post_opt(spank_t sp, int ac, char** av) {
    if (!has_predicttime_arg)
        return 0;

    pid_t srun_pid = getpid();
    char cmdline_file_str[BUFSIZ];
    sprintf(cmdline_file_str, "/proc/%d/cmdline", (int)srun_pid);

    char* cmdline_str = NULL;
    int cmdline_len = 0;

    FILE* cmdline_file = fopen(cmdline_file_str, "r");
    while (!feof(cmdline_file)) {
        cmdline_str = (char*)realloc(cmdline_str, (cmdline_len + BUFSIZ) * sizeof(char));
        cmdline_len += fread(cmdline_str + cmdline_len, 1, BUFSIZ, cmdline_file);
    }
    fclose(cmdline_file);

    int proc_ac = 0;
    char** proc_av = NULL;

    int c = 0;
    for (int i = 0; i < cmdline_len; ++i) {
        if (cmdline_str[i] == '\0') {
            proc_av = (char**)realloc(proc_av, (proc_ac + 1) * sizeof(char*));
            proc_av[proc_ac] = (char*)malloc((i - c + 1) * sizeof(char));
            strcpy(proc_av[proc_ac], cmdline_str + c);
            ++proc_ac;
            c = i + 1;
        }
    }

    free(cmdline_str);

    predicttime(proc_ac, proc_av);

    for (int i = 0; i < proc_ac; ++i)
        free(proc_av[i]);
    free(proc_av);

    _exit (0);
    return 0; // на всякий случай
}
