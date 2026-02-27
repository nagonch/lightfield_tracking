name="lift6dof"

xhost +local:1000 >/dev/null

if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null 2>&1 || true
    docker exec -it -e DISPLAY="$DISPLAY" -w "$(pwd)" "$name" bash
else
    docker run \
        --name "$name" \
        --gpus all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -it \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "$(pwd):$(pwd)" \
        -v /home:/home \
        -v /mnt:/mnt \
        -v /tmp:/tmp \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
        --network=host \
        --ipc=host \
        --user "$(id -u):$(id -g)" \
        -e DISPLAY="$DISPLAY" \
        -w "$(pwd)" \
        lift6dof:latest bash
fi