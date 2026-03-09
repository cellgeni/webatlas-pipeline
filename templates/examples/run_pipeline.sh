export LSB_DEFAULT_USERGROUP=team283
nextflow run webatlas-pipeline/main.nf \
        -params-file conf.yaml \
        -entry Full_pipeline \
        -profile sanger, singularitydocker \
        -resume