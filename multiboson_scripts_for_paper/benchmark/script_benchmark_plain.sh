for chi in 4 6 8 ; do
    for ix in {1..20} ; do  
        echo $ix ; 
        julia --project=. multiboson_scripts_for_paper/benchmark/benchmark_plain.jl $chi $ix ; 
    done
done

