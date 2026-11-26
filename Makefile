# ---- Platform detection ----------------------------------------------------
# Windows (MSYS2/MinGW) sets OS=Windows_NT; otherwise ask uname.
UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifeq ($(OS),Windows_NT)
    PLATFORM := windows
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := linux
endif

# ---- Project layout --------------------------------------------------------
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/NeuralNetwork.cpp
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

CU_SRCS := $(SRC_DIR)/gpuVersion.cu
CU_OBJS := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# ---- Common flags ----------------------------------------------------------

# Extra compiler defines, e.g.  make DEFINES="-DBATCH_SIZE=64"
DEFINES ?=

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -I$(INC_DIR) -MMD -MP $(DEFINES) #-DNDEBUG 
LDFLAGS  :=
LDLIBS   := -lz -lm

# ---- CUDA (optional) -------------------------------------------------------
# Auto-detected via nvcc. Force off with:  make NO_CUDA=1
# Pick GPU arch with:                      make CUDA_ARCH=86   (default: 75)
NVCC      ?= nvcc
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)

ifeq ($(NO_CUDA),1)
    HAVE_NVCC :=
endif

CUDA_ARCH ?= 75

ifeq ($(HAVE_NVCC),1)
    CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC)))
    CXXFLAGS  += -DHAVE_NVCC -I$(CUDA_HOME)/include
    NVCCFLAGS := -std=c++17 -O3 -I$(INC_DIR) -DHAVE_NVCC $(DEFINES) \
                 -arch=sm_$(CUDA_ARCH) -Xcompiler -fopenmp
    LDFLAGS   += -L$(CUDA_HOME)/lib64
    LDLIBS    += -lcudart
    OBJS      += $(CU_OBJS)
endif

# ---- raylib (UI) (optional) -----------------------------------------------------------
# Auto-detected via pkg-config (brew / apt / pacman / MSYS2 all ship raylib.pc).
# Force the UI off with:  make NO_UI=1
# C++ code sees -DHAVE_RAYLIB when the UI is built, -DNO_RAYLIB otherwise.
HAVE_RAYLIB := $(shell pkg-config --exists raylib 2>/dev/null && echo 1)

ifeq ($(NO_UI),1)
    HAVE_RAYLIB :=
endif

ifeq ($(HAVE_RAYLIB),1)
    CXXFLAGS += -DHAVE_RAYLIB $(shell pkg-config --cflags raylib)
    LDLIBS   += $(shell pkg-config --libs raylib)
else
    CXXFLAGS += -DNO_RAYLIB
endif

# ---- Per-platform settings -------------------------------------------------
ifeq ($(PLATFORM),macos)
    ifeq ($(origin CXX),default)
        CXX := clang++
    endif
    # Apple clang needs libomp from Homebrew:  brew install libomp
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LDFLAGS  += -L$(LIBOMP_PREFIX)/lib
    LDLIBS   += -lomp
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -mcpu=apple-m1
    else
        CXXFLAGS += -march=native
    endif
    EXE :=
endif

ifeq ($(PLATFORM),linux)
    ifeq ($(origin CXX),default)
        CXX := g++
    endif
    CXXFLAGS += -fopenmp -march=native #-fopt-info
    LDFLAGS  += -fopenmp
    EXE :=
endif

ifeq ($(PLATFORM),windows)
    # Intended for MSYS2 (MinGW64 shell) with: mingw-w64-x86_64-{gcc,raylib,zlib,pkgconf}
    ifeq ($(origin CXX),default)
        CXX := g++
    endif
    CXXFLAGS += -fopenmp -march=native
    LDFLAGS  += -fopenmp
    EXE := .exe
endif


DEPS := $(OBJS:.o=.d)

TARGET := $(BUILD_DIR)/nn$(EXE)

# Plain `make` builds the program (otherwise the first rule below would win)
.DEFAULT_GOAL := all

# ---- Flags stamp (forces a rebuild when flags/defines change) --------------
FLAGS_FILE := $(BUILD_DIR)/.flags

.PHONY: FORCE
FORCE:

# Rewrites the stamp only when the compiler flags actually differ
$(FLAGS_FILE): FORCE | $(BUILD_DIR)
	@echo '$(CXX) $(CXXFLAGS)' | cmp -s - $@ || echo '$(CXX) $(CXXFLAGS)' > $@

# ---- Targets ---------------------------------------------------------------
.PHONY: all run clean info gpu cpu run-gpu run-cpu

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(FLAGS_FILE) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(FLAGS_FILE) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MF $(@:.o=.d) -MT $@ -c $< -o $@

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

# Run from the project root so the relative data/ paths resolve
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) build-gpu build-cpu

info:
	@echo "Platform: $(PLATFORM)"
	@echo "CXX:      $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS:  $(LDFLAGS)"
	@echo "LDLIBS:   $(LDLIBS)"
	@echo "DEFINES:  $(DEFINES)"
	@echo "raylib:   $(if $(HAVE_RAYLIB),yes,no)"
	@echo "cuda:     $(if $(HAVE_NVCC),yes (sm_$(CUDA_ARCH)),no)"


# make gpu : CUDA build (fails with a clear message if nvcc is missing)
gpu:
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "error: nvcc not found, cannot build the GPU version"; exit 1; }
	@$(MAKE) --no-print-directory all NO_CUDA=0 BUILD_DIR=build-gpu

# make cpu : CPU-only build (no nvcc, no .cu files, no -DHAVE_NVCC)
cpu:
	@$(MAKE) --no-print-directory all NO_CUDA=1 BUILD_DIR=build-cpu

run-gpu: gpu
	./build-gpu/nn$(EXE)

run-cpu: cpu
	./build-cpu/nn$(EXE)

# Header dependency files generated by -MMD
-include $(DEPS)
