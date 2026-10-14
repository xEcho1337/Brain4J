package org.brain4j.math.gpu.device;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.silicon.api.Silicon;
import org.silicon.api.cache.MemoryPool;
import org.silicon.api.cache.Pooled;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.device.ComputeContext;
import org.silicon.api.device.ComputeDevice;
import org.silicon.api.kernel.ComputeQueue;

import org.brain4j.math.tensor.TensorKey;

/**
 * GPU device backed by the Silicon compute API.
 * Use it to allocate buffers and run kernels on one GPU.
 */
public class Device {

    private final ComputeDevice device;
    private final ComputeContext context;
    private final MemoryPool<TensorKey> memoryPool;
    private final List<Pooled> pooledQueue;
    private final ComputeQueue queue;
    private final String name;
    private final int deviceIndex;
    private Thread threadContext;

    /**
     * Opens the default GPU (index 0).
     */
    public Device() {
        this(0);
    }

    /**
     * Opens the GPU at the given index.
     */
    public Device(int deviceIndex) {
        try {
            this.deviceIndex = deviceIndex;
            this.device = Silicon.createDevice(deviceIndex);
            this.context = device.createContext();
            this.threadContext = Thread.currentThread();
            this.queue = context.createQueue();
            this.name = device.name();
            this.memoryPool = context.createPool();
            this.pooledQueue = new ArrayList<>();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create GPU device at index " + deviceIndex, e);
        }
    }

    private void ensureSameThread() {
        Thread current = Thread.currentThread();

        if (current != threadContext) {
            context.syncThread();
            threadContext = current;
        }
    }

    /**
     * Waits for pending work, then releases pooled buffers.
     */
    public synchronized void createResources() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
        }

        if (!pooledQueue.isEmpty()) {
            pooledQueue.forEach(Pooled::close);
            pooledQueue.clear();
        }
    }

    /**
     * Returns a pooled buffer for the given key, allocating on a miss.
     *
     * @return the acquired buffer
     */
    public synchronized ComputeBuffer acquire(TensorKey key, Supplier<ComputeBuffer> allocator) {
        ensureSameThread();

        Pooled pooled = memoryPool.acquire(key, allocator);
        pooledQueue.add(pooled);

        return pooled.value();
    }

    /**
     * Allocates a GPU buffer initialized from the given data.
     *
     * @return the new buffer
     */
    public ComputeBuffer createBuffer(float[] data) {
        ensureSameThread();

        try {
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }

    /**
     * Allocates a GPU buffer initialized from the given data.
     *
     * @return the new buffer
     */
    public ComputeBuffer createBuffer(int[] data) {
        ensureSameThread();

        try {
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array", e);
        }
    }

    /**
     * Allocates an uninitialized GPU buffer of the given byte size.
     *
     * @return the new buffer
     */
    public ComputeBuffer createBuffer(long byteSize) {
        ensureSameThread();

        try {
            return context.allocateBytes(byteSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer of size " + byteSize, e);
        }
    }

    /**
     * Returns the buffer pool for tensor memory.
     *
     * @return the memory pool
     */
    public MemoryPool<TensorKey> memoryPool() {
        return memoryPool;
    }

    /**
     * Returns the device display name.
     *
     * @return the device name
     */
    public String name() {
        return name;
    }

    /**
     * Returns the device index.
     *
     * @return the index passed at construction
     */
    public int deviceIndex() {
        return deviceIndex;
    }

    /**
     * Returns the underlying Silicon device.
     *
     * @return the compute device
     */
    public ComputeDevice device() {
        return device;
    }

    /**
     * Returns the compute context for this device.
     *
     * @return the compute context
     */
    public ComputeContext context() {
        return context;
    }

    /**
     * Returns the command queue for this device.
     *
     * @return the compute queue
     */
    public ComputeQueue queue() {
        return queue;
    }

    /**
     * Waits for pending work, then frees the queue and pool.
     */
    public synchronized void free() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
            queue.free();
        }

        memoryPool.free();
    }

    /**
     * Waits for pending work, then closes pooled buffers.
     */
    public synchronized void closeResources() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
        }

        pooledQueue.forEach(Pooled::close);
        pooledQueue.clear();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "Device{" +
            "backend=Silicon" +
            ", name='" + name + '\'' +
            '}';
    }
}
