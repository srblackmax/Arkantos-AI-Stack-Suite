/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <obsidian/obsidian.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

using namespace obsidian;

namespace
{

void
print_version (void)
{
  std::cout << "obsidian-nlp (Obsidian) " << version_string () << "\n"
            << "Copyright (C) 2026 Kival\n"
            << "License GPLv3+: GNU GPL version 3 or later "
            << "<https://gnu.org/licenses/gpl.html>\n"
            << "This is free software: you are free to change and "
            << "redistribute it.\n"
            << "There is NO WARRANTY, to the extent permitted by law.\n";
}

void
print_help (const char *prog)
{
  std::cout << "Usage: " << prog << " [OPTION]... [COMMAND] [TEXT]...\n"
            << "High-performance C++23 Deep Learning & NLP CLI utility.\n\n"
            << "Options:\n"
            << "  --help       display this help and exit\n"
            << "  --version    output version information and exit\n\n"
            << "Commands:\n"
            << "  info         display SIMD ISA and Basalt backend status\n"
            << "  stem WORD    run the 5-step Martin Porter stemmer on WORD\n"
            << "  annotate TXT run full classical NLP annotation on TXT\n";
}

} /* namespace */

int
main (int argc, char **argv)
{
  if (argc < 2)
    {
      print_help (argv[0]);
      return EXIT_SUCCESS;
    }

  std::string_view arg1 = argv[1];
  if (arg1 == "--version" || arg1 == "-v")
    {
      print_version ();
      return EXIT_SUCCESS;
    }
  if (arg1 == "--help" || arg1 == "-h")
    {
      print_help (argv[0]);
      return EXIT_SUCCESS;
    }

  if (arg1 == "info")
    {
      std::cout << "Obsidian Version : " << version_string () << "\n"
                << "Active SIMD ISA  : " << simd::active_isa_name () << "\n"
                << "Basalt Backend   : CPU + Autograd active\n";
      return EXIT_SUCCESS;
    }

  if (arg1 == "stem")
    {
      if (argc < 3)
        {
          std::cerr << "obsidian-nlp: stem requires a word argument\n";
          return EXIT_FAILURE;
        }
      Porter_Stemmer stemmer;
      std::cout << stemmer.stem (argv[2]) << "\n";
      return EXIT_SUCCESS;
    }

  if (arg1 == "annotate")
    {
      if (argc < 3)
        {
          std::cerr << "obsidian-nlp: annotate requires a text argument\n";
          return EXIT_FAILURE;
        }
      Linguistic_Pipeline pipe;
      auto res = pipe.annotate (argv[2]);
      if (!res.has_value ())
        {
          std::cerr << res.error ().format () << "\n";
          return EXIT_FAILURE;
        }
      std::cout << "Sentences: " << res->sentences.size () << "\n";
      for (size_t i = 0; i < res->tokens.size (); ++i)
        {
          std::cout << "  [" << i << "] token=\"" << res->tokens[i]
                    << "\" stem=\"" << res->stems[i]
                    << "\" lemma=\"" << res->lemmas[i]
                    << "\" pos=" << res->pos_tags[i] << "\n";
        }
      return EXIT_SUCCESS;
    }

  std::cerr << "obsidian-nlp: unrecognized option or command '"
            << arg1 << "'\n";
  return EXIT_FAILURE;
}
