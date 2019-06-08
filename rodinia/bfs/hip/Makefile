CC=hipcc
#DEPS=kernel.h,kernel2.h
#OBJ=bfs_try

bfs_try: bfs.cpp kernel.h kernel2.h
	$(CC) -o $@ $< --amdgpu-target=gfx801

