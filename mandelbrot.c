#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

# define NPOINTS 1000
# define MAXITER 1000

struct d_complex{
   double r; 
   double i;
};

struct d_complex c;
int numoutside = 0;
int i,j;
double area, error, eps  = 1.0e-5;

/* lorsque la fonction de test est sans parametre d'entre/sortie 
   elle fait le traitement sur les variables globales ce qui provoque un conflit dans 
   les threads et ça qui conduit a une erreur de calcules

   1-donc on a defini le nombre complex c comme entrée pour que chaque thread exécute la fonction avec 
     son propre c et le conflit est évité

   2-numoutside est une variable globale qu'on doit l'incrémenter a chaque échec de test, 
     donc on a modifié la fonction pour qu'elle retourne un entier qui prends la valeur 1 lors de l'echec 
     de test sinon il prends la valeur 0, puis le résultat sera envoyé et traité dans le code parallel*/
int testpoint(struct d_complex c){
       struct d_complex z;
       z=c;
       int iter;
       double temp; 
       for (iter=0; iter<MAXITER; iter++){
         temp = (z.r*z.r)-(z.i*z.i)+c.r;
         z.i = z.r*z.i*2+c.i;
         z.r = temp;
         if ((z.r*z.r+z.i*z.i)>4.0) {
           return 1;
         }
       }
       return 0;
}

// il est préférable de mettre le code parallel dans une fonction hors de main pour avoir une bonne visibilité du code
void aire_mandelbort_omp(void){
   /*3-eps est accessible par les threads en lecture seulement donc pas besoin de la mettre en privé
     
     4-chaque thread va prendre une partie de la premiére boucle, par contre la deuxiéme boucle sera 
       executée plusieurs fois par chaque thread indépendament aux autres, donc on déclare la variable j 
       comme variable privé    
   */
   #pragma omp parallel default(shared) private(c,j)
   {
   /*5-numoutside est une variable partagé donc sa modification provoque un conflit alors on crée 
       une variable numoutpriv qui est locale pour chaque thread */
   int numoutpriv = 0;
   #pragma omp for
   for (i=0; i<NPOINTS; i++) {
	 for (j=0; j<NPOINTS; j++) {
       c.r = -2.0 +2.5 * (double)(i) / (double)(NPOINTS) + eps;
       c.i = 1.125 * (double)(j) / (double)(NPOINTS) + eps;
       /*6-aprés avoir calculé le nouveau nombre complex, on le passe au test et on additionne le résultat avec 
	       la variable locale numoutpriv */
       numoutpriv+=testpoint(c);
     }
   }
   /*7-on effectue la somme des numoutpriv déjà  calculés, la somme sera affecter à la variable globale numoutside 
       en utilisant une opération atomic pour éviter les conflits et avoir un maximum de rapidité parceque chaque thread va
       exécuter une seule opération atomique*/
   #pragma omp atomic
     numoutside+=numoutpriv;
   }
}

int main(){
   
   //appel de la fonction qui contient le code parallel
   aire_mandelbort_omp();
   area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
   error=area/(double)NPOINTS;

   printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
   printf("Correct answer should be around 1.510659\n");
}
