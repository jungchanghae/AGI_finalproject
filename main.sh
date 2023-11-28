for candidate_num in 4 5 10
do
    for template_idx in 2 3
    do
        python ./sangbeom/main.py --candidate_num ${candidate_num} --template_idx ${template_idx}
    done
done

for candidate_num in 2 4 5
do
    for template_idx in 0 1 2 3
    do
        python ./sangbeom/main2.py --candidate_num ${candidate_num} --template_idx ${template_idx}
    done
done