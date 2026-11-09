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

#ifndef BELLOWS_CLI_COMMON_HPP
#define BELLOWS_CLI_COMMON_HPP

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows
{

inline void
emit_cli_error (const std::string &program, const std::string &message)
{
  std::cerr << program << ": " << message << "\n";
}

inline std::string
trim_token (const std::string &s)
{
  std::size_t first = s.find_first_not_of (" \t\r\n");
  if (first == std::string::npos)
    {
      return "";
    }
  std::size_t last = s.find_last_not_of (" \t\r\n");
  return s.substr (first, last - first + 1);
}

inline double
parse_cell_token (const std::string &raw_token)
{
  std::string token = trim_token (raw_token);
  if (token.empty () || token == "NA" || token == "nan" || token == "NaN"
      || token == "NULL" || token == "null" || token == "?")
    {
      return missing_value ();
    }
  std::size_t idx = 0;
  double val = std::stod (token, &idx);
  if (idx != token.size ())
    {
      throw std::invalid_argument ("invalid numeric token '" + token + "'");
    }
  return val;
}

inline std::vector<double>
parse_csv_line (const std::string &raw_line)
{
  std::string line = trim_token (raw_line);
  std::stringstream ss (line);
  std::string token;
  std::vector<double> vals;
  while (std::getline (ss, token, ','))
    {
      vals.push_back (parse_cell_token (token));
    }
  if (!line.empty () && line.back () == ',')
    {
      vals.push_back (missing_value ());
    }
  return vals;
}

// Parse a comma-separated tabular file where the last column is the target.
inline std::pair<data_table, std::vector<double>>
load_labeled_csv (const std::string &path)
{
  std::ifstream in (path);
  if (!in)
    {
      throw std::runtime_error ("cannot open input file '" + path + "'");
    }

  std::vector<std::vector<double>> rows;
  std::vector<double> targets;
  std::string line;
  std::size_t line_no = 0;

  while (std::getline (in, line))
    {
      ++line_no;
      std::string trimmed = trim_token (line);
      if (trimmed.empty () || trimmed[0] == '#')
        {
          continue;
        }
      std::vector<double> vals = parse_csv_line (trimmed);
      if (vals.size () < 2)
        {
          throw std::runtime_error (path + ":" + std::to_string (line_no)
                                    + ": expected at least 2 columns");
        }
      double target_val = vals.back ();
      if (is_missing (target_val))
        {
          throw std::runtime_error (path + ":" + std::to_string (line_no)
                                    + ": target value cannot be missing");
        }
      targets.push_back (target_val);
      vals.pop_back ();
      if (!rows.empty () && vals.size () != rows[0].size ())
        {
          throw std::runtime_error (path + ":" + std::to_string (line_no)
                                    + ": inconsistent column count");
        }
      rows.push_back (std::move (vals));
    }

  if (rows.empty ())
    {
      return {data_table (), {}};
    }

  std::size_t n = rows.size ();
  std::size_t d = rows[0].size ();
  data_table table (n, d);
  for (std::size_t r = 0; r < n; ++r)
    {
      for (std::size_t c = 0; c < d; ++c)
        {
          table.at (r, c) = rows[r][c];
        }
    }
  return {table, targets};
}

// Parse an unlabeled comma-separated feature file.
inline data_table
load_unlabeled_csv (const std::string &path)
{
  std::ifstream in (path);
  if (!in)
    {
      throw std::runtime_error ("cannot open input file '" + path + "'");
    }

  std::vector<std::vector<double>> rows;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline (in, line))
    {
      ++line_no;
      std::string trimmed = trim_token (line);
      if (trimmed.empty () || trimmed[0] == '#')
        {
          continue;
        }
      std::vector<double> vals = parse_csv_line (trimmed);
      if (!vals.empty ())
        {
          if (!rows.empty () && vals.size () != rows[0].size ())
            {
              throw std::runtime_error (path + ":" + std::to_string (line_no)
                                        + ": inconsistent column count");
            }
          rows.push_back (std::move (vals));
        }
    }

  if (rows.empty ())
    {
      return data_table ();
    }
  std::size_t n = rows.size ();
  std::size_t d = rows[0].size ();
  data_table table (n, d);
  for (std::size_t r = 0; r < n; ++r)
    {
      for (std::size_t c = 0; c < d; ++c)
        {
          table.at (r, c) = rows[r][c];
        }
    }
  return table;
}

} // namespace bellows

#endif // BELLOWS_CLI_COMMON_HPP
