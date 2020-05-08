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

int main() {
    FILE* cmd = fopen("/proc/14457/cmdline", "r");
    char cmdline[8000];
    int cmdlinelen = fread(cmdline, 1, 8000, cmd);
    for (int i = 0; i < cmdlinelen; ++i)
        if (cmdline[i] == '\0')
            cmdline[i] = ' ';
    printf("%d\n%s\n", cmdlinelen, cmdline);
    return 0;
}

