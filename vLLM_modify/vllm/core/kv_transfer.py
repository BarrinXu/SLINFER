import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
import pickle
import psutil
import os
import time


def kv_sender(send_queue: mp.Queue, logger_queue: mp.Queue):
    logger_queue.put(f'start to send kv')
    while True:
        tensor_config = send_queue.get()
        src_rank = tensor_config['src_rank']
        dst_rank = tensor_config['dst_rank']

        dst_sock_addr = tensor_config['dst_socket_addr']
        dst_sock_port = tensor_config['dst_socket_port']
        logger_queue.put(f'kv_tensor begin to send, info: {tensor_config}')
        tensor_data = send_queue.get()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((dst_sock_addr, dst_sock_port))
            tensor_config_in_bytes = pickle.dumps(tensor_config)
            assert len(tensor_config_in_bytes) < 32768
            s.sendall(tensor_config_in_bytes)
        dist.send(tensor=tensor_data, dst=dst_rank)


def kv_receiver(connection_config, receive_queue: mp.Queue, logger_queue: mp.Queue):
    socket_addr = connection_config['socket_addr']
    socket_port = connection_config['socket_port']

    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == socket_port:
            pid = conn.pid
            if pid:
                logger_queue.put(f"Killing process with PID: {pid} on port {socket_port}")
                os.kill(pid, 9)
    time.sleep(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((socket_addr, socket_port))
        s.listen()
        logger_queue.put(f'start to receive kv')
        while True:
            conn, addr = s.accept()
            with conn:
                data_config = conn.recv(32768)
            data_config = pickle.loads(data_config)
            src_rank = data_config['src_rank']
            tensor_shape = data_config['tensor_shape']
            logger_queue.put(f'kv_tensor begin to receive, info: {data_config}')
            tensor = torch.empty(tensor_shape, dtype=torch.float16, device='cpu')
            tensor.share_memory_()
            dist.recv(tensor=tensor, src=src_rank)
            receive_queue.put(data_config)
            receive_queue.put(tensor)


def kv_transfer_manager(connection_config, send_queue, receive_queue, logger_queue):
    rank = connection_config['rank']
    world_size = connection_config['world_size']
    master_addr = connection_config['master_addr']
    master_port = connection_config['master_port']
    logger_queue.put(
        f'initializing gloo, rank: {rank}, world_size: {world_size}, url: tcp://{master_addr}:{master_port}')
    dist.init_process_group(
        backend='gloo', rank=rank, world_size=world_size, init_method=f'tcp://{master_addr}:{master_port}')
    t1 = threading.Thread(target=kv_sender, args=(send_queue, logger_queue))
    t2 = threading.Thread(target=kv_receiver, args=(connection_config, receive_queue, logger_queue))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
