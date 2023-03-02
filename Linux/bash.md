# 循环执行一个命令
`while [ true ]; do
    python gpt_conclusion.py & { sleep 300; kill $! & }
    sleep 60
done`
