cxx = g++

CFLAGS = -g -fPIC
CFLAGS += `root-config --cflags`

INCLUDES = `root-config --incdir`
INCLUDES += `/usr/bin/python2.7-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`
INCLUDES += -I$(LARLITE_BASEDIR)/../
INCLUDES += `ublarcvapp-config --includes`

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`
LIBS += `ublarcvapp-config --libs` 

LDFLAGS = `root-config --ldflags` -lstdc++
LDFLAGS += `/usr/bin/python2.7-config --ldflags`

LDLIBS = `root-config --libs`
LDLIBS = `/usr/bin/python2.7-config --libs`

EXEC = pytest sparse_full_deploy

all: $(EXEC)
%: %.cxx
		$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@.o
		$(CXX) $(LDFLAGS) -o $@ $@.o $(LDLIBS) $(LIBS)
		rm $@.o

clean:
	rm $(EXEC)
