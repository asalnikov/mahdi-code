#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <slurm/slurm_errno.h>
#include "...../slurmctld/slurmctld.h"
#define BUFFER_SIZE 1000


/* Essential for Slurm job_submit plugin interface */
const char plugin_name[] = "Resource Estimator Plugin";
const char plugin_type[] = "job_submit/ML_plugin";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

/* Global variables */
const char *Plug_name = "job_submit_ML";
const char *target_base = "..../jobscripts";


/* Get current date string in "%F" ("%Y-%m-%d") format. */
int _get_datestr (char *ds, int len) {
    time_t t;
    struct tm *lt;

    /* Get current time */
    t = time(NULL);

    if (t == ((time_t) -1)) {
        info("%s: Unable to get current time", Plug_name);
        return -1;
    }

    /* Convert to local time */
    lt = localtime(&t);

    if (lt == NULL) {
        info("%s: Unable to convert to local time", Plug_name);
        return -1;
    }

    /* Convert to string format */
    if (strftime(ds, len, "%F", lt) == 0) {
        info("%s: Unable to convert to date string", Plug_name);
        return -1;
    }

    return 0;
}


/****************************** TO COPY THE JOBSCRIPT AND SAVE IT ********************************************/

extern int job_submit(struct job_descriptor *job_desc, uint32_t submit_uid,
        char **err_msg) {
    /* The job_desc->job_id is not available at submit time, so there is no way to
            identify the job by job id. The alternative is as follow: */
    uint32_t jobid = job_desc->job_id;

    char ds[11];    /* Date string in "%F" ("%Y-%m-%d") format */
    char target_dir[PATH_MAX];      /* Target directory */
    char target_script[PATH_MAX];   /* Target job script filename */
    char target_workdir[PATH_MAX];  /* Target workdir filename */
    FILE *fd = NULL;    /* File handle for write */
    int rv;

    /* Do not proceed if the jobscript is not available */
    if (job_desc->script == NULL) return SLURM_SUCCESS;

    /* Get the current date string */
    if (_get_datestr(ds, sizeof(ds))) {
        info("%s: Unable to get current date string", Plug_name);
        return ESLURM_INTERNAL;
    }

    /* Construct target directory location to store daily job scripts */
    rv = snprintf(target_dir, PATH_MAX, "%s/%s", target_base, ds);

    if (rv < 0 || rv > PATH_MAX - 1) {
        info("%s: Unable to contruct target directory: %s/%s", Plug_name, target_base,
            ds);
        return ESLURM_INTERNAL;
    }

    /* Construct target script location to save current jobscript. */
    rv = snprintf(target_script, PATH_MAX, "%s/job%lu.script", target_dir,
         jobid);

    if (rv < 0 || rv > PATH_MAX - 1) {
        info("%s: Unable to construct target script: %s/job%lu.script", Plug_name,
            target_dir, jobid);
        return ESLURM_INTERNAL;
    }

    /* Construct target wordir location to save current job workdirectory */
    rv = snprintf(target_workdir, PATH_MAX, "%s/job%lu.workdir", target_dir,
         jobid);

    if (rv < 0 || rv > PATH_MAX - 1) {
        info("%s: Unable to construct target workdirectory: %s/job%lu.workdir",
            Plug_name, target_dir, jobid);
        return ESLURM_INTERNAL;
    }

    /* Ignore if target script exists. */
    if (access(target_script, F_OK) == 0) {
        info("%s: %s exists, ignore", Plug_name, target_script);
        return SLURM_SUCCESS;
    } 

    /* Create target directory to store job scripts. */
    /* If it doesn't exist, create it, otherwise ignore. */
    if (mkdir(target_dir, 0750) && errno != EEXIST) {
        info("%s: Unable to mkdir(%s): %m", Plug_name, target_dir);
        return ESLURM_INTERNAL;
    }

    /* Open the target file for write */
    fd = fopen(target_script, "wb");

    if (fd == NULL) {
        info("%s: Unable to open %s: %m", Plug_name, target_script);
        return ESLURM_INTERNAL;
    }

    /* Write job script to file. */
    fwrite(job_desc->script, strlen(job_desc->script), 1, fd);

    if (ferror(fd)) {
        info("%s: Error on writing %s: %m", Plug_name, target_script);
        return ESLURM_WRITING_TO_FILE;
    }

    /* Close the target file. */
    fclose(fd);

    info("%s: Job script saved as %s", Plug_name, target_script);

    /* Open the target file for write. */
    fd = fopen(target_workdir, "wb");

    if (fd == NULL) {
        info("%s: Unable to open %s: %m", Plug_name, target_workdir);
        return ESLURM_INTERNAL;
    }

    /* Write job workdir to file. */
    fwrite(job_desc->work_dir, strlen(job_desc->work_dir), 1, fd);

    if (ferror(fd)) {
        info("%s: Error on writing %s: %m", Plug_name, target_workdir);
        return ESLURM_WRITING_TO_FILE;
    }

    /* Close the target file */
    fclose(fd);

    info("%s: Job workdir saved as %s", Plug_name, target_workdir);

    return SLURM_SUCCESS;
}



/****************************** CHECKING THE JOBSCRIPT FOR VARIABLES AND VALUES ********************************************/

 /* Open the target file for read */
  fd = fopen(target_workdir, "r");

  if(fd == NULL)
    {
       	/* Unable to open file therefore exit */
       	info("%s: Unable to open %s: %m", Plug_name, target_workdir);
	return ESLURM_INTERNAL;
    }


typedef struct var {
    char *var_name;
    char *value;
 } Var;

    char *p, *pp;
    Var var;


while(fgets(buffer , BUFFER_SIZE, fd)!=NULL){

    if(buffer[0] == '#' || buffer[0] == '\n')
      continue;  /* skip the rest of the loop and continue*/
    pp = NULL;
    p = buffer;

       
/*    // comment will be deleted
    while(NULL!=(p=strstr(p, "##"))){
        pp = p;
        p +=2;
    }
*/    
    //spaces will be deleted
/*    while(NULL!=(p=strstr(p, "\n"))){
        pp = p;
        p +=2;
    }
*/
    if(pp != NULL)
        *pp = '\0';
    else
        pp = strchr(buffer, '\0');
    /*cut the end*/
    while(isspace(pp[-1]==' '))
        *--pp = '\0';
    
    
    
    p=strchr(buffer, '=');
    var.var_name = malloc( p - buffer +1);
    *p='\0';/*split*/
    strcpy(var.var_name, buffer);
    pp = strchr(p, '\0');
    var.value = malloc(pp - p);
    strcpy(var.value, p+1);

    printf("%s=%s", var.var_name, var.value);
}

    fclose(fd); 


extern int job_modify(struct job_descriptor *job_desc,
        struct job_record *job_ptr, uint32_t submit_uid) {
    return SLURM_SUCCESS;
}


