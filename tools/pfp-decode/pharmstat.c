
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 

#define NPHARM 10549
#define MAXLINE 5000
#define NINT 330

main(int argc, char *argv[])
  {
  int   i,j,n,nbits,setbits[NPHARM+1];
  unsigned int dbbits[NINT];
  float avebits,percent;
  char c,pharmlist[NPHARM+1][40];
  char sInline[MAXLINE];
  char *ptok;
  FILE *fp;

  if ( argc<2 ) {
	  printf("Usage: pharmstat <pfpfile> [pharm10549.list]\n");
	  exit(0);
  }
  

  /* The label table used to be looked up only in the current directory, so
     the tool segfaulted on a NULL FILE* when run from anywhere else. It is
     now taken from argv[2], then $PHARM_LIST, then the current directory. */
  {
  const char *listpath = (argc > 2) ? argv[2]
                       : (getenv("PHARM_LIST") ? getenv("PHARM_LIST")
                                               : "pharm10549.list");
  fp = fopen(listpath,"r");
  if ( fp == NULL ) {
    fprintf(stderr,"ERROR: can't open pharmacophore label file %s\n",listpath);
    exit(1);
  }
  }
  for (i = 1; i <= NPHARM; i++)
    {
    j = 0; 
    while ((c = getc(fp)) != '\n') pharmlist[i][j++] = c;
    pharmlist[i][j] = '\0';
    }
  fclose(fp);

  for (i = 0; i <= NPHARM; i++) setbits[i] = 0;
  avebits = 0.0;

  if ( (fp = fopen(argv[1],"r")) == NULL ) {printf("ERROR: can't open input file %s\n",argv[1]);exit(0);}

  n=0;
  for (;;)
    {

	  memset(sInline,(int)'\0',MAXLINE);

		if ( !fgets(sInline, MAXLINE, fp) )	break;
      
		n++;
	  

		ptok = strtok(sInline, " ");	 // ignore first token (i.e. name)
		j = 0;
		do {
			ptok = strtok('\0'," ");
			if ( ptok ) {
				sscanf(ptok,"%u",&dbbits[j]);
				//   printf("%d: %u\n",j,dbbits[j]);
			}
			j++;
			if ( j >= NINT )	break;
		} while ( ptok );




    nbits = 0;
    for (i = 0; i < NPHARM; i++)   
      if ((dbbits[i/32] & (1 << 31-i%32)) > 0) 
        { nbits++; setbits[i+1] = 1; }
    avebits += nbits;
    } 
  /* n-- removed: fgets already stops at EOF, so n is the true record
     count. The old decrement made a single-molecule file report n=0 and an
     infinite average. */

  avebits /= (float)n;
  percent = 0.0;
  for (i = 1; i <= NPHARM; i++) percent += (float)setbits[i];
  percent = percent/(float)NPHARM * 100.0;
  printf("\n");
  printf("%6d = number of bitstrings processed\n",n);
  printf("%6.1f = average number of pharmacophores hit per molecule\n",avebits);
  printf("%6.1f = percentage of pharmacophores hit at least once (out of %d)\n",percent,NPHARM);
  printf("\nPharmacophore hit list:\n");
  printf("\n");
  printf("              p1\n");
  printf("             /  \\\n");
  printf("            /    \\\n");
  printf("       d2  /      \\  d3\n");
  printf("          /        \\\n");
  printf("         /          \\\n");
  printf("        /            \\\n");
  printf("      p3 ------------ p2\n");
  printf("              d1\n");
  printf("\n");
  for (i = 1; i <= NPHARM; i++)
    if (setbits[i] == 1) printf("%5d%s\n",i,pharmlist[i]);
  printf("\n");
  }

