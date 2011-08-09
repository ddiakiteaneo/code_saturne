/*============================================================================
 * Syntax checker program for MEI
 *============================================================================*/

/*
  This file is part of the "Mathematical Expression Interpreter" library.

  Copyright (C) 2008-2010  EDF

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*----------------------------------------------------------------------------
 * BFT library headers
 *----------------------------------------------------------------------------*/

#include <bft_mem.h>
#include <bft_error.h>
#include <bft_sys_info.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "mei_evaluate.h"

/*============================================================================
 * Local type definitions
 *============================================================================*/

typedef enum {

  CS_MEI_SUCCESS,
  CS_MEI_INVALID_INPUT,
  CS_MEI_INVALID_SYNTAX,
  CS_MEI_MISSING_SYMBOLS

} cs_mei_check_retcode_t;

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Main program
 *----------------------------------------------------------------------------*/

int
main(void)
{
  int i;

  size_t  read_size = 0;
  size_t  input_size = 0;
  size_t  max_input_size = 256;
  size_t  input_pos = 0;

  int n_symbols = 0;
  int n_missing = 0;
  int n_required = 0;
  int n_errors = 0;

  int retcode = CS_MEI_SUCCESS;

  mei_tree_t  *t = NULL;

  char  *input = NULL;
  char  *memlog = NULL;
  char  **required_symbols = NULL;

  /* Initialize memory management */

  memlog = getenv("CS_MEI_MEM_LOG");
  bft_mem_init(memlog);

  /* Read input */

  BFT_MALLOC(input, max_input_size, char);

  while (!feof(stdin)) {
    read_size = max_input_size - input_size;
    read_size = fread(input + input_size, 1, read_size, stdin);
    input_size += read_size;
    if (input_size == max_input_size) {
      max_input_size *= 2;
      BFT_REALLOC(input, max_input_size, char);
    }
  }

  /* Convert record separator (ctl-^, ASCII decimal 30) to null-character
     so as to simplify processing */

  for (input_pos = 0; input_pos < input_size; input_pos++) {
    if (input[input_pos] == '\30')
      input[input_pos] = '\0';
  }

  /* Build MEI tree */

  input_pos = 0;

  t = mei_tree_new(input);

  input_pos += strlen(input) + 1;
  if (input_pos > input_size) exit(CS_MEI_INVALID_INPUT);
  n_symbols = atoi(input + input_pos);

  for (i = 0; i < n_symbols; i++) {
    input_pos += strlen(input + input_pos) + 1;
    if (input_pos > input_size) exit(CS_MEI_INVALID_INPUT);
    mei_tree_insert(t, input + input_pos, 0.0);
  }

  input_pos += strlen(input + input_pos) + 1;
  if (input_pos > input_size) exit(CS_MEI_INVALID_INPUT);
  n_required = atoi(input + input_pos);

  if (n_required > 0)
    BFT_MALLOC(required_symbols, n_required, char *);

  for (i = 0; i < n_required; i++) {
    input_pos += strlen(input + input_pos) + 1;
    if (input_pos > input_size) exit(CS_MEI_INVALID_INPUT);
    BFT_MALLOC(required_symbols[i], strlen(input + input_pos) + 1, char);
    strcpy(required_symbols[i], input + input_pos);
  }

  BFT_FREE(input);

  /* Evaluate expression */

  n_errors = mei_tree_builder(t);

  if (n_errors != 0) {
    retcode = CS_MEI_INVALID_SYNTAX;
    fprintf(stderr, "%d\n\30", t->errors);
    for (i = 0; i < t->errors; i++) {
      fprintf(stderr, "%d\30%d\30%s\30",
              t->lines[i], t->columns[i], t->labels[i]);
    }
  }

  /* Test that required symbols are defined */

  if (retcode == CS_MEI_SUCCESS && n_required > 0) {

    n_missing = mei_tree_find_symbols(t,
                                      n_required,
                                      (const char **)required_symbols);
    for (i = 0; i < n_required; i++)
      BFT_FREE(required_symbols[i]);
    BFT_FREE(required_symbols);
    if (n_missing > 0) {
      retcode = CS_MEI_MISSING_SYMBOLS;
      fprintf(stderr, "%d\n\30", t->errors);
      for (i = 0; i < t->errors; i++)
        fprintf(stderr, "%s\30", t->labels[i]);
    }
  }

  /* Finalize memory management and return */

  mei_tree_destroy(t);

  bft_mem_end();
  exit(retcode);

  return(retcode);
}

