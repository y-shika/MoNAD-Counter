#include "MoNADCounter.h"

int main(int argc, char * argv[]) {
	MoNAD monad;

	monad.learn();
	monad.counter();

	system("pause");

	return 0;
}