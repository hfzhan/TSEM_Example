source = $(wildcard *.cpp)
object = $(patsubst %.cpp, %.o, $(source))
path_afepack = /usr/local/include/
path_dealii = /usr/local/include/deal.II
path_lapacke = $(HOME)/Packages/lapack-3.11/LAPACKE/include

CPPFLAGS = -I../include -I/usr/local/include -I$(path_afepack) -I$(path_dealii) -I$(path_lapake)

LIBS = -lAFEPack -ldeal_II -ltbb -llapack

CXXFLAGS = -O2 -g -Wno-deprecated-declarations

CXX = mpicxx


%.o : %.cpp
	$(CXX) -c -o $@ $< -I. $(CXXFLAGS) $(CPPFLAGS)

main : $(object)
	$(CXX) -o $@ $(object) $(LDFLAGS) $(LIBS)

clean :
	-rm -rf $(object)
	-rm -rf main
	-rm -rf *.dx

.PHONY : default clean symbolic
