/* pfp2bits - Reads a pfpfile and creates a format with "1's" and"0's"

	7/17/02	SMM Original - based on pfp2pat
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 

#define NPHARM 10549
#define MAXLINE 10000
#define NINT 330

typedef char STRING[80];

main(int argc, char *argv[])
{
	int   i,j,k,n,nbits;
	unsigned int dbbits[NINT];
	char c;
	char sInline[MAXLINE];
	char *ptok;
	int cs;
	STRING s;
	FILE *fp;

	if ( argc<2 ) {
		printf("Usage: pfp2bits <pfpfile>\n");
		exit(0);
	}


	if ( (fp = fopen(argv[1],"r")) == NULL ) {printf("ERROR: can't open input file %s\n",argv[1]);exit(0);}



	memset(sInline,(int)'\0',MAXLINE);
	while ( fgets(sInline, MAXLINE, fp) ) {


		for ( i=0;i<NINT;i++ ) {
			dbbits[i]=0;
		}

		ptok = strtok(sInline, " ");	 
		fprintf(stdout,"%s ", ptok);    // first field is name
			fprintf(stderr,"%s ", ptok);    // first field is name

		j = 0;
		do {
			ptok = strtok('\0'," ");
			if ( ptok )  {
				sscanf(ptok,"%u",&dbbits[j]);
			  // fprintf(stderr,"%u ", dbbits[j]);
			}
			j++;

			if (j >= NINT) break;

		} while ( ptok );

		// Now process the loaded fingerprint

		cs = 0;

		for ( i = 0; i < NPHARM; i++ ) {
					if ( (dbbits[i/32] & (1 << 31-i%32)) > 0 ) {
						fprintf(stdout,"1");
						cs++;
					} else {
						fprintf(stdout,"0");
					}
		}

		fprintf(stdout,"\n");

		fprintf(stderr,"numbits: %d\n", cs);

		memset(sInline,(int)'\0',MAXLINE);

	}

	fclose(fp);

}






