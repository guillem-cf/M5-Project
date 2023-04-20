# Bash script. Read the file test_task_a_as_image.txt and test_task_a_as_text.txt
# Loop through the file and run the command for each line
echo "Running test_task_a.sh"


for line in $(cat /ghome/group03/M5-Project/week5/Results/test_task_a_as_image.txt)
do
    echo "$line"
    echo loading model "$line"/task_a_triplet_10.pth
    python task_a_inference.py --weights_model="$line"/task_a_triplet_10.pth --dim_out_fc=as_image
done



for line in $(cat /ghome/group03/M5-Project/week5/Results/test_task_a_as_text.txt)
do
    echo "$line"
    echo loading model "$line"/task_a_triplet_10.pth
    python task_a_inference.py --weights_model="$line"/task_a_triplet_10.pth --dim_out_fc=as_text
done
