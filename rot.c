#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
 
int i;

double* rot_omp(int N, double* arr) {
  double* T = malloc((N) * sizeof(double));
  // sauvegarder le premier élement pour éviter son écrasement
  double prem=arr[0];
  
  // décaler parallélement le tableau dans le tableau temporaire
  #pragma omp parallel for
  for (i = 0; i < N - 1; i++)
	T[i]=arr[i+1];
  // récupérer la variable sauvegardé dans le dernier élement du tableau
  T[N-1] = prem;
  
  // retourner le pointeur de tableau décalé
  return T;
}

void rot_omp_sans_tabl(int N, double* arr) {
  int prem=arr[0];  
  #pragma omp parallel private(i)
    {
    	
    	// récupération de l'id et le nombre de threads
        int nbr_th = omp_get_num_threads();
        int num_th = omp_get_thread_num();

        // calcul du l'index du début et la fin des élements parcouru dans le tableau par le thread
        int bloc = (N-2)/nbr_th;
        int deb = num_th*bloc;
        int fin = deb + bloc - 1;
        // mettre l'avant derniére case comme fin pour le dernier thread 
	// pour éviter de dépasser la taille de tableau
        if (num_th == nbr_th-1) fin = N-2;

        // sauvegarder la premiére variable aprés le dernier élément parcequ'elle sera écrasé par le prochain thread
        double tmp = arr[fin+1];

        //mettre une barriére pour s'assurer que les threads sauvegardent les variables avant l'écrasement
	#pragma omp barrier 
	for (i=deb; i<fin; i++)
            arr[i] = arr[i+1];

        // récupérer la variable sauvegardée
        arr[fin] = tmp;
    }
    arr[N-1] = prem; 
}

void rot(int N, double* arr) {
  for (i = 0; i < N - 1; i++) {
    double tmp = arr[i];
    arr[i] = arr[i+1];
    arr[i+1] = tmp;
  }
}

int main(int argc, char ** argv) {
  if (argc != 2) {
    printf("Usage: %s <N>\n", argv[0]);
    return EXIT_SUCCESS;
  }
  /* REMARQUE : lors de l'execution du programme, on a remarqué que le code séquentiel est plus rapide que les codes paralleles lorsque N est petit
                mais si N >= 10^7 alors les deux codes paralleles s'éxecutent en moins du temps, ceci reviens au temps passé par le programme lors 
                de la création des threads et la division du travail  */
  int N = atoi(argv[1]);

  double* a = malloc(N * sizeof(double));
  for (i = 0; i < N; i++) a[i] = i;
  
  //Sequentiel
  double start = omp_get_wtime();
  rot(N, a);
  double end = omp_get_wtime();
  double error = 0;
  for (i = 0; i < N; i++) error += fabs(a[i] - (i + 1) % N);
  printf("Séquentiel :\nelapsed: %f, error: %f\n", end - start, error);
  
  //OpenMP avec tableau temporaire
  for (i = 0; i < N; i++) a[i] = i;
  start = omp_get_wtime();
  a = rot_omp(N, a);
  end = omp_get_wtime();
  error = 0;
  for (i = 0; i < N; i++) error += fabs(a[i] - (i + 1) % N);
  printf("OMP avec tableau :\nelapsed: %f, error: %f\n", end - start, error);
  
  //OpenMP sans tableau temporaire
  for (i = 0; i < N; i++) a[i] = i;
  start = omp_get_wtime();
  rot_omp_sans_tabl(N, a);
  end = omp_get_wtime();
  error = 0;
  for (i = 0; i < N; i++) error += fabs(a[i] - (i + 1) % N);
  printf("OMP sans tableau : \nelapsed: %f, error: %f\n", end - start, error);
}
